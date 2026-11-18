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

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pytest_httpserver import HTTPServer

from nemo_curator.core.vllm_server import VLLMSubprocessServer


class TestVLLMSubprocessServer:
    def test_endpoint_uses_configured_port(self) -> None:
        server = VLLMSubprocessServer(model="some-model", port=9876)
        assert server.endpoint == "http://localhost:9876/v1"

    def test_endpoint_uses_default_port(self) -> None:
        server = VLLMSubprocessServer(model="some-model")
        assert server.endpoint == "http://localhost:8000/v1"

    def test_stop_before_start_is_noop(self) -> None:
        server = VLLMSubprocessServer(model="some-model")
        server.stop()
        assert server._started is False
        assert server._process is None

    def test_stop_is_idempotent(self) -> None:
        server = VLLMSubprocessServer(model="some-model")
        server.stop()
        server.stop()
        assert server._started is False

    def test_command_construction(self) -> None:
        """Verify the vLLM command is built correctly from dataclass fields."""
        server = VLLMSubprocessServer(
            model="Qwen/Qwen3-Omni-30B",
            port=8200,
            host="127.0.0.1",
            python_executable="/usr/bin/python3",
            extra_args=["--dtype", "bfloat16", "-tp", "2"],
        )

        with patch("nemo_curator.core.vllm_server.subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            # Also mock the health check to succeed immediately
            with patch.object(server, "_wait_for_healthy"):
                server.start()

            cmd = mock_popen.call_args[0][0]
            assert cmd == [
                "/usr/bin/python3",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                "Qwen/Qwen3-Omni-30B",
                "--port",
                "8200",
                "--host",
                "127.0.0.1",
                "--dtype",
                "bfloat16",
                "-tp",
                "2",
            ]
            assert mock_popen.call_args[1]["start_new_session"] is True
            assert mock_popen.call_args[1]["shell"] is False

        # Clean up
        server._process = None
        server._started = False

    def test_wait_for_healthy_success(self, httpserver: HTTPServer) -> None:
        """Health check succeeds when /v1/models returns 200."""
        httpserver.expect_request("/v1/models").respond_with_json({"data": []})
        server = VLLMSubprocessServer(
            model="some-model",
            port=httpserver.port,
            health_check_timeout_s=5,
        )
        # Mock the process so poll() returns None (still running)
        server._process = MagicMock()
        server._process.poll.return_value = None

        # Should not raise
        server._wait_for_healthy()

    def test_wait_for_healthy_timeout(self) -> None:
        """Health check raises TimeoutError on unreachable port."""
        server = VLLMSubprocessServer(
            model="some-model",
            port=19876,
            health_check_timeout_s=2,
        )
        # Mock the process so poll() returns None (still running)
        server._process = MagicMock()
        server._process.poll.return_value = None

        with pytest.raises(TimeoutError, match="did not become ready within 2s"):
            server._wait_for_healthy()

    def test_premature_exit_detection(self) -> None:
        """Health check raises RuntimeError if vLLM exits before becoming healthy."""
        server = VLLMSubprocessServer(
            model="some-model",
            port=19876,
            health_check_timeout_s=10,
        )
        # Mock a process that has already exited with code 1
        server._process = MagicMock()
        server._process.poll.return_value = 1
        server._process.returncode = 1

        with pytest.raises(RuntimeError, match="exited prematurely with return code 1"):
            server._wait_for_healthy()

    def test_context_manager(self) -> None:
        """Context manager calls start/stop."""
        server = VLLMSubprocessServer(model="some-model")
        with patch.object(server, "start") as mock_start, patch.object(server, "stop") as mock_stop:
            with server:
                mock_start.assert_called_once()
            mock_stop.assert_called_once()

    def test_log_file_opens_and_closes(self, tmp_path: Path) -> None:
        """When log_file is set, stdout is redirected and file handle is cleaned up."""
        log_path = str(tmp_path / "vllm.log")
        server = VLLMSubprocessServer(
            model="some-model",
            log_file=log_path,
        )

        with patch("nemo_curator.core.vllm_server.subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            with patch.object(server, "_wait_for_healthy"):
                server.start()

            # Verify Popen was called with stdout=file and stderr=STDOUT
            call_kwargs = mock_popen.call_args[1]
            assert call_kwargs["stdout"] is not None
            assert call_kwargs["stderr"].__class__.__name__ == "int"  # subprocess.STDOUT is -2

        # Log file handle should be open
        assert server._log_fh is not None

        # Stop should close it
        server._process = None  # prevent actual kill
        server.stop()
        assert server._log_fh is None
