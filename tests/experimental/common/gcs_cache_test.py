# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for GCS JAX compilation cache utilities."""

import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
from unittest import mock

from absl.testing import absltest
from google.api_core import exceptions as google_exceptions
from tunix.experimental.common import datatypes
from tunix.experimental.common import gcs_cache
from tunix.experimental.orchestrator import orchestrator
from tunix.experimental.orchestrator import rl_program
from tunix.experimental.worker import mock_worker
from tunix.experimental.worker import remote_execution


def _blob(name: str) -> mock.Mock:
  blob = mock.Mock()
  blob.name = name
  return blob


class GcsCacheTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.enter_context(mock.patch.dict(os.environ, {}, clear=True))

  def _gcs_modules(
      self, mock_storage: mock.MagicMock, mock_tm: mock.MagicMock
  ) -> dict[str, mock.MagicMock]:
    mock_storage.transfer_manager = mock_tm
    mock_cloud = mock.MagicMock()
    mock_cloud.storage = mock_storage
    return {
        "google.cloud": mock_cloud,
        "google.cloud.storage": mock_storage,
        "google.cloud.storage.transfer_manager": mock_tm,
    }

  def test_parse_gcs_uri(self) -> None:
    bucket, prefix = gcs_cache._parse_gcs_uri("gs://my-bucket/path/to/cache")
    self.assertEqual(bucket, "my-bucket")
    self.assertEqual(prefix, "path/to/cache/")

    bucket, prefix = gcs_cache._parse_gcs_uri("gs://my-bucket")
    self.assertEqual(bucket, "my-bucket")
    self.assertEqual(prefix, "")

    with self.assertRaises(ValueError):
      gcs_cache._parse_gcs_uri("https://not-gcs.com")

  def test_ensure_jax_cache_env(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      target_dir = pathlib.Path(tmpdir) / "test_cache"
      res = gcs_cache.ensure_jax_cache_env(target_dir)
      self.assertEqual(res, target_dir)
      self.assertTrue(target_dir.is_dir())
      self.assertEqual(os.environ["JAX_COMPILATION_CACHE_DIR"], str(target_dir))
      self.assertEqual(os.environ["VLLM_XLA_CACHE_PATH"], str(target_dir))

  def test_ensure_jax_cache_env_updates_jax_config(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      target_dir = pathlib.Path(tmpdir) / "test_cache"
      mock_jax = mock.MagicMock()
      with mock.patch.dict(sys.modules, {"jax": mock_jax}):
        res = gcs_cache.ensure_jax_cache_env(target_dir)
        self.assertEqual(res, target_dir)
        mock_jax.config.update.assert_called_once_with(
            "jax_compilation_cache_dir", str(target_dir)
        )

  @mock.patch.object(gcs_cache, "download_cache", return_value=True)
  def test_restore_jax_cache_with_explicit_uri(
      self, mock_download: mock.MagicMock
  ) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      cache_dir = pathlib.Path(tmpdir) / "cache"
      success = gcs_cache.restore_jax_cache(
          gcs_uri="gs://bucket/rollout_cache",
          local_dir=cache_dir,
      )
      self.assertTrue(success)
      mock_download.assert_called_once_with(
          cache_dir, "gs://bucket/rollout_cache"
      )

  @mock.patch.object(gcs_cache, "download_cache", return_value=True)
  def test_restore_jax_cache_from_env(
      self, mock_download: mock.MagicMock
  ) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      cache_dir = pathlib.Path(tmpdir) / "cache"
      with mock.patch.dict(
          os.environ, {"ROLLOUT_JAX_CACHE_GCS_DIR": "gs://bucket/rollout_env"}
      ):
        success = gcs_cache.restore_jax_cache(local_dir=cache_dir)
        self.assertTrue(success)
        mock_download.assert_called_once_with(
            cache_dir, "gs://bucket/rollout_env"
        )

  def test_save_jax_cache_disabled(self) -> None:
    with mock.patch.dict(os.environ, {"SAVE_JAX_CACHE": "false"}):
      success = gcs_cache.save_jax_cache(gcs_uri="gs://bucket/test")
      self.assertFalse(success)

  @mock.patch.object(gcs_cache, "upload_cache", return_value=True)
  def test_save_jax_cache_success(self, mock_upload: mock.MagicMock) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      cache_dir = pathlib.Path(tmpdir) / "cache"
      cache_dir.mkdir(parents=True, exist_ok=True)
      with mock.patch.dict(os.environ, {"SAVE_JAX_CACHE": "true"}):
        success = gcs_cache.save_jax_cache(
            gcs_uri="gs://bucket/saved_cache",
            local_dir=cache_dir,
        )
        self.assertTrue(success)
        mock_upload.assert_called_once_with(
            cache_dir, "gs://bucket/saved_cache"
        )

  def test_upload_cache_skip_if_exists(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      cache_dir = pathlib.Path(tmpdir) / "cache"
      cache_dir.mkdir()
      (cache_dir / "obj1").write_text("dummy")
      (cache_dir / "obj2").write_text("dummy")

      mock_storage = mock.MagicMock()
      mock_tm = mock.MagicMock()
      precondition_failed = google_exceptions.PreconditionFailed(
          "412 Precondition Failed"
      )
      mock_tm.upload_many_from_filenames.return_value = [
          precondition_failed,
          None,
      ]

      with mock.patch.dict(
          sys.modules, self._gcs_modules(mock_storage, mock_tm)
      ):
        success = gcs_cache.upload_cache(cache_dir, "gs://test-bucket/prefix")
        self.assertTrue(success)
        mock_tm.upload_many_from_filenames.assert_called_once_with(
            mock.ANY,
            mock.ANY,
            source_directory=str(cache_dir),
            blob_name_prefix="prefix/",
            skip_if_exists=True,
            worker_type=mock_tm.THREAD,
            max_workers=mock.ANY,
        )

  def test_upload_cache_skips_transfer_when_all_in_gcs(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      cache_dir = pathlib.Path(tmpdir) / "cache"
      (cache_dir / "sub").mkdir(parents=True)
      (cache_dir / "obj1").write_text("dummy")
      (cache_dir / "sub" / "obj2").write_text("dummy")

      mock_storage = mock.MagicMock()
      mock_tm = mock.MagicMock()
      client = mock_storage.Client.return_value
      client.list_blobs.return_value = [
          _blob("prefix/obj1"),
          _blob("prefix/sub/obj2"),
      ]
      with mock.patch.dict(
          sys.modules, self._gcs_modules(mock_storage, mock_tm)
      ):
        self.assertTrue(gcs_cache.upload_cache(cache_dir, "gs://b/prefix"))
      mock_tm.upload_many_from_filenames.assert_not_called()

  def test_upload_cache_uploads_only_missing_with_threads(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      cache_dir = pathlib.Path(tmpdir) / "cache"
      cache_dir.mkdir()
      (cache_dir / "obj1").write_text("dummy")
      (cache_dir / "obj2").write_text("dummy")

      mock_storage = mock.MagicMock()
      mock_tm = mock.MagicMock()
      client = mock_storage.Client.return_value
      client.list_blobs.return_value = [_blob("prefix/obj1")]
      mock_tm.upload_many_from_filenames.return_value = [None]
      with mock.patch.dict(
          sys.modules, self._gcs_modules(mock_storage, mock_tm)
      ):
        self.assertTrue(gcs_cache.upload_cache(cache_dir, "gs://b/prefix"))
      args, kwargs = mock_tm.upload_many_from_filenames.call_args
      self.assertEqual(args[1], ["obj2"])
      self.assertIs(kwargs["worker_type"], mock_tm.THREAD)

  def test_download_cache_uses_thread_worker_type(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      cache_dir = pathlib.Path(tmpdir) / "cache"
      mock_storage = mock.MagicMock()
      mock_tm = mock.MagicMock()
      client = mock_storage.Client.return_value
      client.list_blobs.return_value = [_blob("prefix/obj1")]
      mock_tm.download_many_to_path.return_value = [None]
      with mock.patch.dict(
          sys.modules, self._gcs_modules(mock_storage, mock_tm)
      ):
        self.assertTrue(
            gcs_cache.download_cache(cache_dir, "gs://test-bucket/prefix")
        )
      mock_tm.download_many_to_path.assert_called_once_with(
          mock.ANY,
          ["obj1"],
          destination_directory=str(cache_dir),
          blob_name_prefix="prefix/",
          worker_type=mock_tm.THREAD,
          max_workers=mock.ANY,
      )

  @mock.patch.object(gcs_cache, "save_jax_cache", return_value=True)
  def test_worker_upload_jax_cache_delegates_to_save_jax_cache(
      self, mock_save: mock.MagicMock
  ) -> None:
    worker = mock_worker.MockWorker(
        worker_id="rollout-0", roles=frozenset({"rollout"})
    )
    self.assertTrue(
        worker.upload_jax_cache(
            gcs_uri="gs://bucket/rollout_cache", local_dir="/tmp/cache"
        )
    )
    mock_save.assert_called_once_with(
        gcs_uri="gs://bucket/rollout_cache",
        local_dir="/tmp/cache",
    )

  def test_orchestrator_sync_jax_cache(self) -> None:
    orch = orchestrator.ClusterOrchestrator(
        jax_cache_config=gcs_cache.JaxCacheConfig(
            save_jax_cache=True,
            rollout_jax_cache_gcs_dir="gs://bucket/orch_rollout",
        )
    )
    mock_handle = mock.MagicMock(spec=remote_execution.ActorHandle)
    orch.register_worker_handle(
        "rollout-0",
        roles=[datatypes.Role.ROLLOUT],
        handle=mock_handle,
    )
    orch.sync_jax_cache()
    mock_handle.submit.assert_called_once_with(
        "upload_jax_cache", gcs_uri="gs://bucket/orch_rollout"
    )

  def test_orchestrator_sync_jax_cache_rollout_only(self) -> None:
    orch = orchestrator.ClusterOrchestrator(
        jax_cache_config=gcs_cache.JaxCacheConfig(
            save_jax_cache=True,
            rollout_jax_cache_gcs_dir="gs://bucket/orch_rollout",
        )
    )
    mock_rollout = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_trainer = mock.MagicMock(spec=remote_execution.ActorHandle)
    orch.register_worker_handle(
        "rollout-0",
        roles=[datatypes.Role.ROLLOUT],
        handle=mock_rollout,
    )
    orch.register_worker_handle(
        "trainer-0",
        roles=[datatypes.Role.ACTOR],
        handle=mock_trainer,
    )
    orch.sync_jax_cache()
    mock_rollout.submit.assert_called_once_with(
        "upload_jax_cache", gcs_uri="gs://bucket/orch_rollout"
    )
    mock_trainer.submit.assert_not_called()

  def test_orchestrator_sync_jax_cache_single_worker(self) -> None:
    orch = orchestrator.ClusterOrchestrator(
        jax_cache_config=gcs_cache.JaxCacheConfig(
            save_jax_cache=True,
            rollout_jax_cache_gcs_dir="gs://bucket/orch_rollout",
        )
    )
    mock_rollout_0 = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_rollout_0.submit.return_value = True
    mock_rollout_1 = mock.MagicMock(spec=remote_execution.ActorHandle)
    orch.register_worker_handle(
        "rollout-0",
        roles=[datatypes.Role.ROLLOUT],
        handle=mock_rollout_0,
    )
    orch.register_worker_handle(
        "rollout-1",
        roles=[datatypes.Role.ROLLOUT],
        handle=mock_rollout_1,
    )
    orch.sync_jax_cache()
    mock_rollout_0.submit.assert_called_once_with(
        "upload_jax_cache", gcs_uri="gs://bucket/orch_rollout"
    )
    mock_rollout_1.submit.assert_not_called()

  def test_orchestrator_sync_jax_cache_primary_failure_does_not_fallback(
      self,
  ) -> None:
    orch = orchestrator.ClusterOrchestrator(
        jax_cache_config=gcs_cache.JaxCacheConfig(
            save_jax_cache=True,
            rollout_jax_cache_gcs_dir="gs://bucket/orch_rollout",
            sync_timeout_s=45.0,
        )
    )
    mock_rollout_0 = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_rollout_0.submit.side_effect = RuntimeError("GCS upload failed")

    mock_rollout_1 = mock.MagicMock(spec=remote_execution.ActorHandle)

    orch.register_worker_handle(
        "rollout-0",
        roles=[datatypes.Role.ROLLOUT],
        handle=mock_rollout_0,
    )
    orch.register_worker_handle(
        "rollout-1",
        roles=[datatypes.Role.ROLLOUT],
        handle=mock_rollout_1,
    )
    orch.sync_jax_cache()
    mock_rollout_0.submit.assert_called_once_with(
        "upload_jax_cache", gcs_uri="gs://bucket/orch_rollout"
    )
    mock_rollout_1.submit.assert_not_called()

  def test_orchestrator_sync_jax_cache_bounds_hung_upload(self) -> None:
    orch = orchestrator.ClusterOrchestrator(
        jax_cache_config=gcs_cache.JaxCacheConfig(
            save_jax_cache=True,
            rollout_jax_cache_gcs_dir="gs://bucket/orch_rollout",
            sync_timeout_s=0.2,
        )
    )
    release = threading.Event()
    hung = lambda *a, **k: release.wait()
    mock_rollout_0 = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_rollout_0.submit.side_effect = hung
    mock_rollout_1 = mock.MagicMock(spec=remote_execution.ActorHandle)
    orch.register_worker_handle(
        "rollout-0", roles=[datatypes.Role.ROLLOUT], handle=mock_rollout_0
    )
    orch.register_worker_handle(
        "rollout-1", roles=[datatypes.Role.ROLLOUT], handle=mock_rollout_1
    )
    try:
      start = time.monotonic()
      orch.sync_jax_cache()
      self.assertLess(time.monotonic() - start, 5.0)
      # Abandoned uploads must not block interpreter exit.
      threads = {t.name: t for t in threading.enumerate()}
      self.assertTrue(threads["jax-cache-upload-rollout-0"].daemon)
    finally:
      release.set()
    mock_rollout_0.submit.assert_called_once()
    mock_rollout_1.submit.assert_not_called()

  def test_orchestrator_sync_jax_cache_uses_worker_registered_uri(self) -> None:
    orch = orchestrator.ClusterOrchestrator(
        jax_cache_config=gcs_cache.JaxCacheConfig(save_jax_cache=True)
    )
    mock_handle = mock.MagicMock(spec=remote_execution.ActorHandle)
    orch.register_worker_handle(
        "rollout-0",
        roles=[datatypes.Role.ROLLOUT],
        handle=mock_handle,
        resources={"jax_cache_gcs_dir": "gs://bucket/derived_rollout"},
    )
    orch.sync_jax_cache()
    mock_handle.submit.assert_called_once_with(
        "upload_jax_cache", gcs_uri="gs://bucket/derived_rollout"
    )

  def test_orchestrator_run_triggers_post_step0_sync(self) -> None:
    orch = orchestrator.ClusterOrchestrator(
        jax_cache_config=gcs_cache.JaxCacheConfig(
            save_jax_cache=True,
            rollout_jax_cache_gcs_dir="gs://bucket/orch_rollout",
        )
    )
    mock_handle = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_handle.submit.return_value = True
    orch.register_worker_handle(
        "rollout-0",
        roles=[datatypes.Role.ROLLOUT],
        handle=mock_handle,
    )

    class _FakeProgram(rl_program.RLProgram):

      def run(self, engine: object, **kwargs: object) -> None:
        del engine, kwargs
        if self.on_step_end is not None:
          self.on_step_end(0, None)
          self.on_step_end(1, None)

    program = _FakeProgram()
    orch.run(program)
    orch.shutdown()
    # upload_jax_cache fires twice: once after bring_up_workers() and once
    # after step 0 (and not after step 1).
    upload_calls = [
        call
        for call in mock_handle.submit.call_args_list
        if call.args and call.args[0] == "upload_jax_cache"
    ]
    self.assertLen(upload_calls, 2)
    for call in upload_calls:
      self.assertEqual(
          call,
          mock.call("upload_jax_cache", gcs_uri="gs://bucket/orch_rollout"),
      )

  def test_classify_tpu_hardware(self) -> None:
    self.assertEqual(gcs_cache.classify_tpu_hardware("tpuv5e:2x4"), "v5e")
    self.assertEqual(
        gcs_cache.classify_tpu_hardware("tpu-v5-lite-podslice:2x4"), "v5e"
    )
    self.assertEqual(gcs_cache.classify_tpu_hardware("tpuv5:2x2x1"), "v5p")
    self.assertEqual(gcs_cache.classify_tpu_hardware("tpuv5p:2x2x2"), "v5p")
    self.assertEqual(gcs_cache.classify_tpu_hardware("tpuv6e:2x4"), "v6e")
    self.assertEqual(gcs_cache.classify_tpu_hardware("tpu7x:2x2x1"), "v7x")

  def test_derive_rollout_cache_uri_and_fingerprint(self) -> None:
    params_base = gcs_cache.RolloutCacheKeyParams(
        model_name="qwen3.5-35b-a3b",
        sampler="vllm",
        max_prompt_length=512,
        max_response_length=1024,
        tpu_slice="tpuv5:2x2x1",
        mesh_tp=1,
        mesh_fsdp=1,
        mesh_expert=8,
    )
    uri1 = gcs_cache.derive_rollout_cache_uri(
        bucket="gs://test-bucket", params=params_base, env={}
    )
    self.assertIsNotNone(uri1)
    assert uri1 is not None
    self.assertTrue(
        uri1.startswith(
            "gs://test-bucket/jax_cache/v5p/qwen3.5-35b-a3b/rollout_2x2x1_ep8_tp1_"
        )
    )

    # Verify that changing quantization or max_response_length changes the hash.
    params_fp8 = gcs_cache.RolloutCacheKeyParams(
        model_name="qwen3.5-35b-a3b",
        sampler="vllm",
        max_prompt_length=512,
        max_response_length=1024,
        tpu_slice="tpuv5:2x2x1",
        mesh_tp=1,
        mesh_fsdp=1,
        mesh_expert=8,
        rollout_fp8="true",
    )
    uri2 = gcs_cache.derive_rollout_cache_uri(
        bucket="gs://test-bucket", params=params_fp8, env={}
    )
    self.assertNotEqual(uri1, uri2)

    params_longer = gcs_cache.RolloutCacheKeyParams(
        model_name="qwen3.5-35b-a3b",
        sampler="vllm",
        max_prompt_length=512,
        max_response_length=2048,
        tpu_slice="tpuv5:2x2x1",
        mesh_tp=1,
        mesh_fsdp=1,
        mesh_expert=8,
    )
    uri3 = gcs_cache.derive_rollout_cache_uri(
        bucket="gs://test-bucket", params=params_longer, env={}
    )
    self.assertNotEqual(uri1, uri3)

    # Verify JAX_CACHE_GCS_DIR acts as a base prefix and appends rollout
    # subdirectory + digest.
    uri_base_dir = gcs_cache.derive_rollout_cache_uri(
        params=params_base,
        env={"JAX_CACHE_GCS_DIR": "gs://test-bucket/custom_root/"},
    )
    self.assertIsNotNone(uri_base_dir)
    assert uri_base_dir is not None
    self.assertTrue(
        uri_base_dir.startswith(
            "gs://test-bucket/custom_root/v5p/qwen3.5-35b-a3b/rollout_2x2x1_ep8_tp1_"
        )
    )

    # Verify MAXTEXT_OUTPUT_DIR preserves tenant subpath prefixes.
    uri_tenant = gcs_cache.derive_rollout_cache_uri(
        params=params_base,
        env={"MAXTEXT_OUTPUT_DIR": "gs://shared-bucket/tenant_a/exp1/"},
    )
    self.assertIsNotNone(uri_tenant)
    assert uri_tenant is not None
    self.assertTrue(
        uri_tenant.startswith(
            "gs://shared-bucket/tenant_a/exp1/jax_cache/v5p/qwen3.5-35b-a3b/rollout_2x2x1_ep8_tp1_"
        )
    )

  def test_orchestrator_sync_jax_cache_wait_false_does_not_block_on_prior(
      self,
  ) -> None:
    orch = orchestrator.ClusterOrchestrator(
        jax_cache_config=gcs_cache.JaxCacheConfig(
            save_jax_cache=True,
            rollout_jax_cache_gcs_dir="gs://bucket/orch_rollout",
        )
    )
    first_entered = threading.Event()
    release_first = threading.Event()
    call_count = 0

    def _submit(method: str, **kwargs: object) -> bool:
      del kwargs
      nonlocal call_count
      if method == "upload_jax_cache":
        call_count += 1
        if call_count == 1:
          first_entered.set()
          release_first.wait(timeout=5.0)
      return True

    mock_handle = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_handle.submit.side_effect = _submit
    orch.register_worker_handle(
        "rollout-0", roles=[datatypes.Role.ROLLOUT], handle=mock_handle
    )

    f1 = orch.sync_jax_cache(wait=False)
    self.assertIsNotNone(f1)
    self.assertTrue(first_entered.wait(timeout=2.0))

    start = time.monotonic()
    f2 = orch.sync_jax_cache(wait=False)
    elapsed = time.monotonic() - start
    self.assertIsNotNone(f2)
    self.assertLess(elapsed, 0.5)

    release_first.set()
    orch.shutdown()
    assert f1 is not None and f2 is not None
    self.assertTrue(f1.result(timeout=2.0))
    self.assertTrue(f2.result(timeout=2.0))
    self.assertEqual(call_count, 2)

  def test_orchestrator_sync_jax_cache_local_worker_async(self) -> None:
    orch = orchestrator.ClusterOrchestrator(
        jax_cache_config=gcs_cache.JaxCacheConfig(
            save_jax_cache=True,
            rollout_jax_cache_gcs_dir="gs://bucket/local_rollout",
        )
    )
    worker = mock_worker.MockWorker(
        worker_id="rollout-local-0", roles=frozenset({"rollout"})
    )
    orch.register_worker(worker)
    with mock.patch.object(
        worker, "upload_jax_cache", return_value=True
    ) as mock_upload:
      fut = orch.sync_jax_cache(wait=False)
      self.assertIsNotNone(fut)
      assert fut is not None
      self.assertTrue(fut.result(timeout=2.0))
      mock_upload.assert_called_once_with(gcs_uri="gs://bucket/local_rollout")

  @mock.patch.object(subprocess, "run")
  def test_run_cli_rsync_handles_timeout(
      self, mock_run: mock.MagicMock
  ) -> None:
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["gsutil"], timeout=1)
    self.assertFalse(
        gcs_cache._run_cli_rsync(["gsutil", "-m", "rsync"], timeout_s=1.0)
    )
    mock_run.assert_called_once_with(
        ["gsutil", "-m", "rsync"], check=False, timeout=1.0
    )

  @mock.patch.object(gcs_cache, "download_cache", return_value=True)
  def test_main_download_derives_uri_when_empty(
      self, mock_download: mock.MagicMock
  ) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      with mock.patch.dict(
          os.environ,
          {
              "JAX_CACHE_BUCKET": "gs://derived-bucket",
              "ROLLOUT_TPU_SLICE": "tpuv5e:2x2",
              "MODEL_NAME": "qwen3-0.6b",
          },
      ):
        with self.assertRaises(SystemExit) as cm:
          gcs_cache.main(["download", tmpdir, ""])
        self.assertEqual(cm.exception.code, 0)
        self.assertEqual(mock_download.call_count, 1)
        called_dir, called_uri = mock_download.call_args.args[:2]
        self.assertEqual(called_dir, pathlib.Path(tmpdir))
        self.assertTrue(
            called_uri.startswith(
                "gs://derived-bucket/jax_cache/v5e/qwen3-0.6b/rollout_2x2_"
            )
        )


if __name__ == "__main__":
  absltest.main()
