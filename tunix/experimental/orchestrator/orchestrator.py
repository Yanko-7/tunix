# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cluster Infrastructure Coordinator (orchestrator.py) following Orchestrator V2.

Supervises WorkerRegistry, LifecycleDriver, HealthMonitor, and StartupValidator.
Provides supervised RL program execution (`run`).
"""

import collections
from collections.abc import Callable, Sequence
from concurrent import futures
import contextlib
import pickle
import threading
import time
from typing import Any, Mapping

from absl import logging
from tunix.experimental.common import datatypes
from tunix.experimental.common import gcs_cache
from tunix.experimental.orchestrator import distributed_rl_engine
from tunix.experimental.orchestrator import health_monitor
from tunix.experimental.orchestrator import lifecycle
from tunix.experimental.orchestrator import rl_program
from tunix.experimental.orchestrator import startup_validation
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.trajectory import store as trajectory_store_lib
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.worker import abstract_worker
from tunix.experimental.worker import remote_execution


_STOP_TIMEOUT_S = 60.0  # Timeout for stopping remote workers. 60 should not be touched for any healthy stop.


class ClusterOrchestrator:
  """Supervises cluster hardware, health monitoring, and program execution."""

  def __init__(
      self,
      config: Any = None,
      registry: worker_registry.WorkerRegistry | None = None,
      lifecycle_driver: lifecycle.LifecycleDriver | None = None,
      monitor: health_monitor.HealthMonitor | None = None,
      weight_sync_mode: str | None = None,
      trajectory_store_config: Mapping[str, Any] | None = None,
      jax_cache_config: gcs_cache.JaxCacheConfig | None = None,
  ):
    """Initializes ClusterOrchestrator.

    Args:
      config: Orchestrator configuration.
      registry: Worker registry to use; one is created if omitted.
      lifecycle_driver: Lifecycle driver to use; one is created if omitted.
      monitor: Health monitor to use; one is created if omitted.
      weight_sync_mode: Weight sync mode, if any.
      trajectory_store_config: Trajectory Store configuration for this process,
        or None to run without a store. See `store.TrajectoryStore.from_config`.
        Pass the same config to every process in the run: for the file backend
        it is the shared root_dir and run_id that will make the workers' writes
        visible to this process's reads once read/write wiring is connected.
      jax_cache_config: Optional typed JAX compilation cache configuration.
        Defaults to `gcs_cache.JaxCacheConfig.from_env()`.
    """
    self.config = config
    self.registry = registry or worker_registry.WorkerRegistry()
    self.lifecycle_driver = lifecycle_driver or lifecycle.LifecycleDriver(
        self.registry
    )
    self.monitor = monitor or health_monitor.HealthMonitor(self.registry)
    self._remote_worker_handles: dict[
        str, list[remote_execution.ActorHandle]
    ] = collections.defaultdict(list)
    self._remote_worker_handles_by_id: dict[
        str, remote_execution.ActorHandle
    ] = {}
    self._remote_worker_infos: dict[str, datatypes.WorkerInfo] = {}
    self.engine: distributed_rl_engine.DistributedRLEngine | None = None
    mode = getattr(weight_sync_mode, "value", weight_sync_mode)
    self._weight_sync_mode = str(mode).lower() if mode is not None else None
    self.jax_cache_config: gcs_cache.JaxCacheConfig = (
        jax_cache_config
        if jax_cache_config is not None
        else gcs_cache.JaxCacheConfig.from_env()
    )
    self._pending_jax_cache_sync: (
        tuple[futures.Future[bool], str, float] | None
    ) = None
    # The sole construction site for this process's Trajectory Store: one
    # ClusterOrchestrator exists per orchestrator process, so building it
    # here — once, in __init__ — is the whole guard. Its lifetime is meant
    # to span the process, not any one run() call, so it is public
    # (`self.trajectory_store`, not `_trajectory_store`) for a caller to
    # thread into whatever RLProgram it constructs; see StandardRLProgram's
    # `trajectory_store` argument.
    # TODO(sizhi): Wire active trajectory reads/writes between
    # orchestrator/program and rollout workers in follow-up CLs.
    store_config = (
        {
            trajectory_store_lib.METADATA_TYPE_KEY: (
                trajectory_lib.TunixTrajectoryMetadata.METADATA_TYPE
            ),
            **trajectory_store_config,
        }
        if trajectory_store_config is not None
        else None
    )
    self.trajectory_store = trajectory_store_lib.TrajectoryStore.from_config(
        store_config
    )
    if self.trajectory_store is not None:
      # Logged so a config mismatch between this process and its workers is one
      # grep away.
      logging.info(
          "[trajectory-store] orchestrator built %s",
          self.trajectory_store.to_redacted_config(),
      )

  def __enter__(self) -> "ClusterOrchestrator":
    """Interactive context manager bring-up."""
    self.bring_up_workers()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    self.shutdown()

  def register_worker_from_hostname(
      self,
      hostname: str,
      _: int,
      metadata: bytes,
      rpc_timeout_s: float = 1800.0,
  ) -> None:
    """Registers a remote worker handle from a hostname and metadata."""
    md = pickle.loads(metadata)

    # NB: this should align with workers
    service_type = md["service_type"]
    service_address = f"{hostname}:{md['service_port']}"
    worker_id = md["worker_id"]

    logging.info(
        "Discovered %s service (%s) at %s.",
        service_type,
        worker_id,
        service_address,
    )

    match service_type:
      case "trainer":
        role = datatypes.Role.ACTOR
      case "rollout":
        role = datatypes.Role.ROLLOUT
      case "inference":
        role = datatypes.Role.REFERENCE
      case _:
        raise RuntimeError(f"unknown service type {service_type}")

    worker_resources: dict[str, str] = {"address": service_address}
    if "jax_cache_gcs_dir" in md and md["jax_cache_gcs_dir"]:
      worker_resources["jax_cache_gcs_dir"] = str(md["jax_cache_gcs_dir"])

    self.register_worker_handle(
        worker_id=worker_id,
        roles=[role],
        handle=remote_execution.ActorHandle.from_address(
            f"grpc://{service_address}",
            rpc_timeout_s=rpc_timeout_s,
        ),
        resources=worker_resources,
    )

  def register_worker(
      self, worker: abstract_worker.Worker
  ) -> datatypes.WorkerInfo:
    """Registers a worker in the WorkerRegistry."""
    return self.registry.register(worker)

  def register_worker_handle(
      self,
      worker_id: str,
      roles: Sequence[datatypes.Role | str],
      handle: remote_execution.ActorHandle,
      resources: dict[str, Any] | None = None,
  ) -> datatypes.WorkerInfo:
    """Registers a remote worker handle used directly by DistributedRLEngine."""
    if not roles:
      raise ValueError(f"worker {worker_id!r} declares no roles")
    if not isinstance(handle, remote_execution.ActorHandle):
      raise TypeError(
          "register_worker_handle expects a remote_execution.ActorHandle, got "
          f"{type(handle)}"
      )
    if (
        worker_id in self._remote_worker_infos
        or worker_id in self.registry.worker_ids()
    ):
      raise ValueError(f"duplicate worker_id: {worker_id!r}")
    role_names = frozenset(
        role.value if isinstance(role, datatypes.Role) else role
        for role in roles
    )
    info = datatypes.WorkerInfo(
        worker_id=worker_id,
        roles=role_names,
        resources={"remote": True, **dict(resources or {})},
    )
    for role in role_names:
      self._remote_worker_handles[role].append(handle)
    self._remote_worker_handles_by_id[worker_id] = handle
    self._remote_worker_infos[worker_id] = info
    logging.info(
        "Registered remote worker %r with roles %s.",
        worker_id,
        sorted(role_names),
    )
    return info

  def unregister_worker(self, worker_id: str) -> None:
    """Unregisters a worker by its id."""
    if worker_id in self._remote_worker_infos:
      info = self._remote_worker_infos.pop(worker_id)
      handle = self._remote_worker_handles_by_id.pop(worker_id)
      for role in info.roles:
        handles = self._remote_worker_handles.get(role)
        if handles is not None:
          self._remote_worker_handles[role] = [
              h for h in handles if h is not handle
          ]
          if not self._remote_worker_handles[role]:
            del self._remote_worker_handles[role]
      return
    self.registry.unregister(worker_id)

  def wait_for_workers(
      self,
      min_workers: dict[datatypes.Role | str, int],
      timeout: float | None = None,
      poll_interval_s: float = 0.5,
  ) -> None:
    """Waits for registered workers to meet the minimum required counts.

    Args:
      min_workers: A dictionary mapping Role or role name to the minimum number
        of workers required.
      timeout: Maximum duration to wait in seconds before raising TimeoutError.
        If None, waits indefinitely until requirements are met.
      poll_interval_s: Time in seconds between polling attempts.

    Raises:
      TimeoutError: If the required worker counts are not met within timeout.
    """
    start_time = time.monotonic()
    while True:
      current_counts = {
          role: len(self.worker_handles(role)) for role in min_workers
      }
      if all(
          current_counts[role] >= target_count
          for role, target_count in min_workers.items()
      ):
        logging.info(
            "All required workers are ready. Current counts: %s",
            current_counts,
        )
        return

      if timeout is not None and (time.monotonic() - start_time) >= timeout:
        raise TimeoutError(
            f"Timed out after {timeout}s waiting for workers. "
            f"Required: {min_workers}, Current: {current_counts}"
        )

      sleep_duration = poll_interval_s
      if timeout is not None:
        remaining = timeout - (time.monotonic() - start_time)
        sleep_duration = min(poll_interval_s, max(0.0, remaining))

      time.sleep(sleep_duration)

  def worker_infos(self) -> list[datatypes.WorkerInfo]:
    """Returns local and remote worker metadata registered with the orchestrator."""
    registry_ids = self.registry.worker_ids()
    return self.registry.infos() + [
        self._remote_worker_infos[worker_id]
        for worker_id in sorted(self._remote_worker_infos)
        if worker_id not in registry_ids
    ]

  def worker_handles(
      self, role: datatypes.Role | str
  ) -> list[remote_execution.ActorHandle]:
    """Returns handles for all workers (remote and local) registered under the given role."""
    return self._get_actor_handles(role)

  def _resolve_rollout_gcs_uri(
      self, primary_worker_id: str | None = None
  ) -> str | None:
    """Resolves the target rollout GCS compilation cache URI, if configured."""
    if self.jax_cache_config.rollout_jax_cache_gcs_dir:
      return self.jax_cache_config.rollout_jax_cache_gcs_dir
    if (
        primary_worker_id is not None
        and primary_worker_id in self._remote_worker_infos
    ):
      resources = self._remote_worker_infos[primary_worker_id].resources
      if "jax_cache_gcs_dir" in resources and resources["jax_cache_gcs_dir"]:
        return str(resources["jax_cache_gcs_dir"])
    for w_id in sorted(self._remote_worker_infos):
      info = self._remote_worker_infos[w_id]
      if (
          datatypes.Role.ROLLOUT.value in info.roles
          and "jax_cache_gcs_dir" in info.resources
          and info.resources["jax_cache_gcs_dir"]
      ):
        return str(info.resources["jax_cache_gcs_dir"])
    if self.jax_cache_config.jax_cache_gcs_dir:
      return self.jax_cache_config.jax_cache_gcs_dir
    return None

  def _await_jax_cache_upload(
      self, pending: tuple[futures.Future[bool], str, float]
  ) -> None:
    """Waits for a specific JAX cache upload future and logs its outcome."""
    outcome, primary_worker_id, sync_timeout_s = pending
    try:
      uploaded = outcome.result(timeout=sync_timeout_s)
    except futures.TimeoutError:
      logging.warning(
          "JAX cache upload on worker %s timed out after %.0fs; abandoning it.",
          primary_worker_id,
          sync_timeout_s,
      )
      return
    except Exception as err:  # pylint: disable=broad-except
      logging.warning(
          "Failed to sync JAX cache on worker %s: %r", primary_worker_id, err
      )
      return
    if uploaded is not None and not uploaded:
      logging.warning(
          "Worker %s reported a failed JAX cache upload.", primary_worker_id
      )
      return
    logging.info("Worker %s JAX cache upload finished.", primary_worker_id)

  def _wait_for_jax_cache_sync(self) -> None:
    """Waits for any in-flight background JAX cache upload to finish."""
    if self._pending_jax_cache_sync is None:
      return
    pending = self._pending_jax_cache_sync
    self._pending_jax_cache_sync = None
    self._await_jax_cache_upload(pending)

  def sync_jax_cache(self, *, wait: bool = True) -> futures.Future[bool] | None:
    """Synchronizes JAX compilation cache from the primary rollout worker to GCS.

    Args:
      wait: If True, blocks until the upload completes (or times out). If False,
        dispatches the upload on a background daemon thread and returns its
        Future immediately so rollout/training critical paths are not blocked.

    Returns:
      The upload Future when a rollout worker upload is launched, or None if
      cache persistence is disabled or inapplicable.
    """
    if (
        not self.jax_cache_config.save_jax_cache
        or gcs_cache.is_jax_cache_disabled()
    ):
      return None

    worker_ids = sorted(self._remote_worker_infos)
    upload_fn: Callable[[], bool]
    if not worker_ids:
      local_rollout_workers = [
          worker
          for worker in self.registry.workers()
          if datatypes.Role.ROLLOUT.value in worker.info().roles
      ]
      if not local_rollout_workers:
        return None
      local_worker = local_rollout_workers[0]
      primary_worker_id = local_worker.info().worker_id
      rollout_gcs_uri = self._resolve_rollout_gcs_uri()
      if not rollout_gcs_uri:
        return None
      upload_fn = lambda: local_worker.upload_jax_cache(gcs_uri=rollout_gcs_uri)
    else:
      rollout_worker_ids = [
          w_id
          for w_id in worker_ids
          if datatypes.Role.ROLLOUT.value
          in self._remote_worker_infos[w_id].roles
      ]
      if not rollout_worker_ids:
        return None
      primary_worker_id = rollout_worker_ids[0]
      rollout_gcs_uri = self._resolve_rollout_gcs_uri(primary_worker_id)
      if not rollout_gcs_uri:
        return None
      handle = self._remote_worker_handles_by_id[primary_worker_id]
      upload_fn = lambda: handle.submit(
          "upload_jax_cache", gcs_uri=rollout_gcs_uri
      )

    logging.info(
        "Triggering JAX compilation cache synchronization to GCS (%s) from"
        " rollout worker %s (wait=%s)...",
        rollout_gcs_uri,
        primary_worker_id,
        wait,
    )

    prior_pending = self._pending_jax_cache_sync
    self._pending_jax_cache_sync = None
    outcome: futures.Future[bool] = futures.Future()

    def _run() -> None:
      if prior_pending is not None:
        self._await_jax_cache_upload(prior_pending)
      try:
        outcome.set_result(upload_fn())
      except Exception as err:  # pylint: disable=broad-except
        outcome.set_exception(err)

    sync_timeout_s = self.jax_cache_config.sync_timeout_s
    # Daemon thread: unlike ThreadPoolExecutor workers it is not joined at
    # interpreter exit, so an abandoned hung RPC (bounded only by the RPC
    # deadline, which can be hours) cannot block process exit.
    threading.Thread(
        target=_run,
        name=f"jax-cache-upload-{primary_worker_id}",
        daemon=True,
    ).start()
    self._pending_jax_cache_sync = (outcome, primary_worker_id, sync_timeout_s)
    if wait:
      self._wait_for_jax_cache_sync()
    return outcome

  def bring_up_workers(self, dummy_data: Any = None) -> None:
    """Brings up all registered workers through lifecycle initialization."""
    logging.info(
        "Bringing up %d registered worker(s)...",
        len(self.worker_infos()),
    )
    self.lifecycle_driver.bring_up(dummy_data)
    self._bring_up_remote_workers(dummy_data)
    self.sync_jax_cache(wait=False)
    self.engine = self._create_engine()
    logging.info("All workers brought up successfully.")

  def shutdown(self) -> None:
    """Shuts down all workers and closes health monitoring resources."""
    logging.info("Shutting down all workers...")
    with contextlib.ExitStack() as stack:
      # Registered in reverse order of execution (LIFO) so that every stage
      # runs even if a preceding stage raises an exception.
      if self.trajectory_store is not None:
        stack.callback(self.trajectory_store.close)
      stack.callback(self.lifecycle_driver.shutdown)
      stack.callback(self._shutdown_remote_workers)
      stack.callback(self._wait_for_jax_cache_sync)
      stack.callback(self.monitor.close)
    logging.info("Shutdown complete.")

  def validate_startup(self, alg_config: Any, training_config: Any) -> None:
    """Validates cluster geometry against configurations."""
    startup_validation.validate_startup(
        self.registry, alg_config, training_config
    )

  def _get_role_members(self, role: datatypes.Role | str) -> list[Any]:
    role_key = role.value if isinstance(role, datatypes.Role) else role
    members = self.registry.group(role_key).members()

    # Fallback in case workers were registered with the enum object directly
    if not members and isinstance(role, datatypes.Role):
      members = self.registry.group(role).members()
    return members

  def _get_actor_handles(
      self, role: datatypes.Role | str
  ) -> list[remote_execution.ActorHandle]:
    role_key = role.value if isinstance(role, datatypes.Role) else role
    handles = list(self._remote_worker_handles.get(role_key, ()))
    handles.extend(
        remote_execution.InProcessActorHandle(
            remote_execution.InProcessRemoteExecutionServer(worker)
        )
        for worker in self._get_role_members(role)
    )
    return handles

  def _bring_up_remote_workers(self, dummy_data: Any = None) -> None:
    """Runs lifecycle hooks concurrently across remote worker handles."""
    worker_ids = sorted(self._remote_worker_infos)
    if not worker_ids:
      return

    # Each worker_id is an independent slice/replica; cross-worker weight-sync
    # pairing runs via WeightSyncCoordinator after bring_up_workers() returns.
    def _bring_up_worker(worker_id: str) -> None:
      handle = self._remote_worker_handles_by_id[worker_id]
      logging.info("Initializing remote worker %s.", worker_id)
      handle.submit("initialize")
      logging.info("Compiling remote worker %s.", worker_id)
      handle.submit("compile", dummy_data)
      logging.info("Starting remote worker %s.", worker_id)
      handle.submit("start")

    failures: list[tuple[str, BaseException]] = []
    with futures.ThreadPoolExecutor(max_workers=len(worker_ids)) as pool:
      fut_to_wid = {
          pool.submit(_bring_up_worker, wid): wid for wid in worker_ids
      }
      for fut in futures.as_completed(fut_to_wid):
        wid = fut_to_wid[fut]
        try:
          fut.result()
        except Exception as err:  # pylint: disable=broad-except
          logging.error("Remote worker %s failed bring-up: %r", wid, err)
          failures.append((wid, err))

    if failures:
      failures.sort(key=lambda item: item[0])
      raise lifecycle.LifecycleError("bring_up", failures) from failures[0][1]

  def _shutdown_remote_workers(self) -> None:
    """Stops remote worker handles best-effort, with a hard timeout."""
    pool = futures.ThreadPoolExecutor(
        max_workers=max(1, len(self._remote_worker_infos))
    )
    stops = {
        worker_id: pool.submit(
            self._remote_worker_handles_by_id[worker_id].submit, "stop"
        )
        for worker_id in sorted(self._remote_worker_infos)
    }
    for worker_id, fut in stops.items():
      try:
        fut.result(timeout=_STOP_TIMEOUT_S)
      except Exception as err:  # pylint: disable=broad-except
        logging.warning("Failed to stop remote worker %s: %r", worker_id, err)
    pool.shutdown(wait=False)

  def _create_engine(self) -> distributed_rl_engine.DistributedRLEngine:
    """Constructs a DistributedRLEngine from the registered role groups."""
    rollout_workers = self._get_actor_handles(datatypes.Role.ROLLOUT)
    actor_workers = self._get_actor_handles(datatypes.Role.ACTOR)
    critic_workers = self._get_actor_handles(datatypes.Role.CRITIC)
    reference_workers = self._get_actor_handles(datatypes.Role.REFERENCE)

    trainer_workers = {}
    if actor_workers:
      trainer_workers[datatypes.Role.ACTOR] = actor_workers[0]
    if critic_workers:
      trainer_workers[datatypes.Role.CRITIC] = critic_workers[0]

    inference_workers = {}
    if reference_workers:
      inference_workers[datatypes.Role.REFERENCE] = reference_workers[0]

    coordinator = None
    if self._weight_sync_mode not in (None, "none"):
      from tunix.experimental.weight_sync import weight_sync_coordinator

      handler = weight_sync_coordinator.create_default_handler(
          mode=self._weight_sync_mode
      )

      handle_to_id = {
          v: k for k, v in self._remote_worker_handles_by_id.items()
      }
      for role, handles in [
          (datatypes.Role.ACTOR, actor_workers),
          (datatypes.Role.ROLLOUT, rollout_workers),
      ]:
        for h in handles:
          w_id = handle_to_id.get(h, f"local-{role.value}-{id(h)}")
          info = self._remote_worker_infos.get(w_id) or datatypes.WorkerInfo(
              worker_id=w_id, roles=frozenset({role.value})
          )
          self.registry.register(weight_sync_coordinator.RemoteWorkerShim(h, info), override=True)  # pyrefly: ignore[bad-argument-type]

      coordinator = weight_sync_coordinator.WeightSyncCoordinator(
          registry=self.registry,
          handler=handler,
          controller_id="auto-coordinator",
      )

    return distributed_rl_engine.DistributedRLEngine(
        rollout_workers=rollout_workers,
        trainer_workers=trainer_workers,
        inference_workers=inference_workers,
        weight_sync_coordinator=coordinator,
    )

  def run(
      self,
      program: rl_program.RLProgram,
      bring_up: bool = True,
      dummy_data: Any = None,
      **kwargs: Any,
  ) -> None:
    """Runs an RL program to completion under supervision.

    Args:
      program: The RL program instance to execute.
      bring_up: Whether to bring up registered workers before execution.
      dummy_data: Optional initialization data passed to worker compilation.
      **kwargs: Additional keyword arguments forwarded to program.run.
    """
    if bring_up:
      self.bring_up_workers(dummy_data=dummy_data)

    self.monitor.poll()
    logging.info("Executing program %s...", type(program).__name__)
    engine = self.engine or self._create_engine()

    should_sync_after_first_step = (
        self.jax_cache_config.save_jax_cache
        and not gcs_cache.is_jax_cache_disabled()
        and self._resolve_rollout_gcs_uri() is not None
    )
    if not should_sync_after_first_step:
      program.run(
          engine=engine,
          **kwargs,
      )
      logging.info("Program %s finished.", type(program).__name__)
      return

    orig_on_step_end = program.on_step_end
    first_step_synced = False

    def _on_step_end_with_cache_sync(step: int, step_result: Any) -> None:
      nonlocal first_step_synced
      if orig_on_step_end is not None:
        orig_on_step_end(step, step_result)
      if not first_step_synced:
        first_step_synced = True
        self.sync_jax_cache(wait=False)

    program.on_step_end = _on_step_end_with_cache_sync
    try:
      program.run(
          engine=engine,
          **kwargs,
      )
    finally:
      program.on_step_end = orig_on_step_end
    if not first_step_synced:
      self.sync_jax_cache(wait=False)
    logging.info("Program %s finished.", type(program).__name__)
