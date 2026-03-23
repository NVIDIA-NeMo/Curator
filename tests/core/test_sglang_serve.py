# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest
from pytest_httpserver import HTTPServer

pytest.importorskip("sglang", reason="sglang not installed")

from nemo_curator.core.sglang_serve import (
    SGLangInferenceServer,
    SGLangModelConfig,
    _active_sglang_servers,
    get_active_sglang_server,
    is_sglang_active,
)

INTEGRATION_TEST_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"  # pragma: allowlist secret


# ---------------------------------------------------------------------------
# SGLangModelConfig unit tests
# ---------------------------------------------------------------------------


class TestSGLangModelConfig:
    def test_default_fields(self) -> None:
        config = SGLangModelConfig(model_path="some/model")
        assert config.dp_size == 1
        assert config.tp_size == 1
        assert config.nnodes == 1
        assert config.dist_port == 50000
        assert config.mem_fraction_static is None
        assert config.gpu_ids is None
        assert config.server_kwargs == {}
        assert config.model_name is None

    def test_build_command_singlenode(self) -> None:
        config = SGLangModelConfig(model_path="meta-llama/Llama-3-8B", tp_size=2, dp_size=4)
        cmd = config._build_command(port=30000)
        assert sys.executable == cmd[0]
        assert "sglang.launch_server" in cmd
        assert "--model-path" in cmd
        assert "meta-llama/Llama-3-8B" in cmd
        assert "--tp-size" in cmd
        assert "2" in cmd
        assert "--dp-size" in cmd
        assert "4" in cmd
        assert "--port" in cmd
        assert "30000" in cmd
        # Single-node: no --nnodes or --node-rank
        assert "--nnodes" not in cmd
        assert "--node-rank" not in cmd

    def test_build_command_multinode_rank0(self) -> None:
        config = SGLangModelConfig(model_path="some/model", nnodes=2)
        cmd = config._build_command(port=30000, node_rank=0, dist_init_addr="10.0.0.1:50000")
        assert "--nnodes" in cmd
        assert "2" in cmd
        assert "--node-rank" in cmd
        assert "0" in cmd
        assert "--dist-init-addr" in cmd
        assert "10.0.0.1:50000" in cmd
        assert "--port" in cmd

    def test_build_command_multinode_worker(self) -> None:
        config = SGLangModelConfig(model_path="some/model", nnodes=2)
        cmd = config._build_command(port=30000, node_rank=1, dist_init_addr="10.0.0.1:50000")
        assert "--node-rank" in cmd
        assert "1" in cmd
        # Worker nodes do not bind the HTTP port
        assert "--port" not in cmd

    def test_build_command_model_name(self) -> None:
        config = SGLangModelConfig(model_path="some/model", model_name="my-model")
        cmd = config._build_command(port=30000)
        assert "--served-model-name" in cmd
        assert "my-model" in cmd

    def test_build_command_mem_fraction_static(self) -> None:
        config = SGLangModelConfig(model_path="some/model", mem_fraction_static=0.8)
        cmd = config._build_command(port=30000)
        assert "--mem-fraction-static" in cmd
        assert "0.8" in cmd

    def test_build_command_server_kwargs(self) -> None:
        config = SGLangModelConfig(
            model_path="some/model",
            server_kwargs={"max_running_requests": 64, "enable_cache_report": True},
        )
        cmd = config._build_command(port=30000)
        assert "--max-running-requests" in cmd
        assert "64" in cmd
        assert "--enable-cache-report" in cmd
        # Bool True flag: no value appended
        assert cmd[cmd.index("--enable-cache-report") + 1] != "True"


# ---------------------------------------------------------------------------
# SGLangInferenceServer unit tests (mocked subprocess)
# ---------------------------------------------------------------------------


class TestSGLangInferenceServer:
    def test_endpoint_uses_configured_port(self) -> None:
        server = SGLangInferenceServer(model=SGLangModelConfig(model_path="m"), port=12345)
        assert server.endpoint == "http://localhost:12345/v1"

    def test_stop_before_start_is_noop(self) -> None:
        server = SGLangInferenceServer(model=SGLangModelConfig(model_path="m"))
        server.stop()
        assert server._started is False

    def test_stop_is_idempotent(self) -> None:
        server = SGLangInferenceServer(model=SGLangModelConfig(model_path="m"))
        server.stop()
        server.stop()
        assert server._started is False

    def test_wait_for_healthy_success(self, httpserver: HTTPServer) -> None:
        httpserver.expect_request("/v1/models").respond_with_json({"data": []})
        server = SGLangInferenceServer(
            model=SGLangModelConfig(model_path="m"),
            port=httpserver.port,
            health_check_timeout_s=5,
        )
        # Should not raise
        server._wait_for_healthy()

    def test_wait_for_healthy_timeout(self) -> None:
        server = SGLangInferenceServer(
            model=SGLangModelConfig(model_path="m"),
            port=19877,
            health_check_timeout_s=2,
        )
        with pytest.raises(TimeoutError, match="did not become ready within 2s"):
            server._wait_for_healthy()

    def test_start_raises_on_port_conflict(self) -> None:
        server = SGLangInferenceServer(model=SGLangModelConfig(model_path="m"), port=30001)
        _active_sglang_servers.add(30001)
        try:
            with pytest.raises(RuntimeError, match="already active on this port"):
                server.start()
        finally:
            _active_sglang_servers.discard(30001)

    def test_verbose_false_sets_logging_env(self) -> None:
        server = SGLangInferenceServer(model=SGLangModelConfig(model_path="m"), verbose=False)
        env = server._build_env()
        assert env["SGLANG_LOGGING_LEVEL"] == "WARNING"

    def test_verbose_true_omits_logging_env(self) -> None:
        server = SGLangInferenceServer(model=SGLangModelConfig(model_path="m"), verbose=True)
        env = server._build_env()
        assert "SGLANG_LOGGING_LEVEL" not in env

    def test_gpu_ids_sets_cuda_visible_devices(self) -> None:
        server = SGLangInferenceServer(
            model=SGLangModelConfig(model_path="m", gpu_ids=[2, 3]),
        )
        env = server._build_env()
        assert env["CUDA_VISIBLE_DEVICES"] == "2,3"

    def test_gpu_ids_not_set_for_multinode(self) -> None:
        """gpu_ids has no effect on multi-node (Ray controls GPU assignment via num_gpus)."""
        server = SGLangInferenceServer(
            model=SGLangModelConfig(model_path="m", gpu_ids=[0, 1], nnodes=2),
        )
        env = server._build_env()
        assert "CUDA_VISIBLE_DEVICES" not in env

    def test_complementary_gpu_ids(self) -> None:
        server = SGLangInferenceServer(
            model=SGLangModelConfig(model_path="m", gpu_ids=[0, 1]),
        )
        assert server.complementary_gpu_ids(4) == [2, 3]

    def test_complementary_gpu_ids_none(self) -> None:
        server = SGLangInferenceServer(model=SGLangModelConfig(model_path="m"))
        assert server.complementary_gpu_ids(4) == [0, 1, 2, 3]

    def test_stop_sends_sigterm_then_sigkill(self) -> None:
        """stop() on a started single-node server terminates the subprocess."""
        import signal

        server = SGLangInferenceServer(model=SGLangModelConfig(model_path="m"), port=30002)
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None
        server._process = mock_proc
        server._started = True
        _active_sglang_servers.add(30002)

        try:
            with patch("os.killpg") as mock_killpg, patch("os.getpgid", return_value=12345):
                server.stop()
            # SIGTERM is the first signal
            first_call_signal = mock_killpg.call_args_list[0][0][1]
            assert first_call_signal == signal.SIGTERM
        finally:
            _active_sglang_servers.discard(30002)

        assert server._started is False
        assert get_active_sglang_server() is None

    def test_start_sets_active_server_state(self, httpserver: HTTPServer) -> None:
        """start() updates _active_sglang_servers and get_active_sglang_server()."""
        httpserver.expect_request("/v1/models").respond_with_json({"data": []})
        server = SGLangInferenceServer(
            model=SGLangModelConfig(model_path="m"),
            port=httpserver.port,
            health_check_timeout_s=5,
        )

        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None  # still alive

        with patch("subprocess.Popen", return_value=mock_proc):
            server.start()

        try:
            assert server._started is True
            assert is_sglang_active()
            assert get_active_sglang_server() is server
        finally:
            # Clean up without calling real stop() to avoid signal issues
            _active_sglang_servers.discard(httpserver.port)
            from nemo_curator.core import sglang_serve

            sglang_serve._active_server = None
            server._started = False

    def test_subprocess_exits_before_healthy_raises_runtime_error(self) -> None:
        """If the subprocess dies before health check passes, raise RuntimeError (not TimeoutError)."""
        server = SGLangInferenceServer(
            model=SGLangModelConfig(model_path="m"),
            port=19878,
            health_check_timeout_s=10,
        )
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = 1  # process has exited
        server._process = mock_proc

        with pytest.raises(RuntimeError, match="exited with code"):
            server._wait_for_healthy()

    def test_context_manager_calls_start_and_stop(self) -> None:
        server = SGLangInferenceServer(model=SGLangModelConfig(model_path="m"))
        with patch.object(server, "start") as mock_start, patch.object(server, "stop") as mock_stop:
            with server:
                mock_start.assert_called_once()
            mock_stop.assert_called_once()


# ---------------------------------------------------------------------------
# Integration tests — real SGLang subprocess
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def sglang_server() -> SGLangInferenceServer:
    """Start SGLangInferenceServer once for all integration tests.

    Uses a small model with enforce_eager-equivalent flags to minimise startup time.
    """
    config = SGLangModelConfig(
        model_path=INTEGRATION_TEST_MODEL,
        dp_size=1,
        tp_size=1,
        server_kwargs={"mem_fraction_static": "0.5"},
    )
    server = SGLangInferenceServer(model=config, health_check_timeout_s=300)
    server.start()

    yield server

    server.stop()


@pytest.mark.gpu
@pytest.mark.usefixtures("sglang_server")
class TestSGLangInferenceServerIntegration:
    """Full lifecycle tests against a real SGLangInferenceServer."""

    def test_is_active_and_queryable(self, sglang_server: SGLangInferenceServer) -> None:
        """Server is active, lists models, and responds to chat completions."""
        from openai import OpenAI

        assert is_sglang_active()
        assert sglang_server._started is True

        client = OpenAI(base_url=sglang_server.endpoint, api_key="na")

        model_ids = [m.id for m in client.models.list()]
        assert INTEGRATION_TEST_MODEL in model_ids

        response = client.chat.completions.create(
            model=INTEGRATION_TEST_MODEL,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=16,
            temperature=0.0,
        )
        assert len(response.choices) > 0
        assert len(response.choices[0].message.content) > 0

    def test_second_start_rejected_same_port(self, sglang_server: SGLangInferenceServer) -> None:
        """Cannot start a second SGLangInferenceServer on the same port."""
        server2 = SGLangInferenceServer(
            model=SGLangModelConfig(model_path=INTEGRATION_TEST_MODEL),
            port=sglang_server.port,
            health_check_timeout_s=60,
        )
        with pytest.raises(RuntimeError, match="already active on this port"):
            server2.start()

        # First server is still healthy and unaffected
        from openai import OpenAI

        client = OpenAI(base_url=sglang_server.endpoint, api_key="na")
        assert INTEGRATION_TEST_MODEL in {m.id for m in client.models.list()}

    def test_restart_after_stop(self, sglang_server: SGLangInferenceServer) -> None:
        """A new SGLangInferenceServer starts cleanly after the previous one is stopped.

        This test must run last — it stops the shared fixture's server.
        """
        from openai import OpenAI

        sglang_server.stop()
        assert not is_sglang_active()

        config = SGLangModelConfig(
            model_path=INTEGRATION_TEST_MODEL,
            dp_size=1,
            tp_size=1,
            server_kwargs={"mem_fraction_static": "0.5"},
        )
        server2 = SGLangInferenceServer(model=config, health_check_timeout_s=300)
        server2.start()

        try:
            client = OpenAI(base_url=server2.endpoint, api_key="na")
            assert INTEGRATION_TEST_MODEL in {m.id for m in client.models.list()}

            response = client.chat.completions.create(
                model=INTEGRATION_TEST_MODEL,
                messages=[{"role": "user", "content": "Say hello in one word."}],
                max_tokens=16,
                temperature=0.0,
            )
            assert len(response.choices[0].message.content) > 0
        finally:
            server2.stop()
