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

import os
import pathlib
from unittest.mock import patch

import yaml

from nemo_curator.metrics.constants import (
    DEFAULT_NEMO_CURATOR_METRICS_PATH,
    GRAFANA_PID_FILE,
    PROMETHEUS_PID_FILE,
    PROMETHEUS_PORT_FILE,
    PROMETHEUS_YAML_TEMPLATE,
)
from nemo_curator.metrics.utils import (
    _get_all_discovery_paths,
    _is_process_running_from_pidfile,
    add_ray_prometheus_metrics_service_discovery,
    get_prometheus_port,
    is_grafana_running,
    is_prometheus_running,
    remove_ray_prometheus_metrics_service_discovery,
)


class TestDefaultMetricsPath:
    def test_default_path_contains_uid(self) -> None:
        """DEFAULT_NEMO_CURATOR_METRICS_PATH should contain the current user's UID."""
        uid = os.getuid()
        assert str(uid) in DEFAULT_NEMO_CURATOR_METRICS_PATH
        assert f"/tmp/nemo_curator_metrics_{uid}" == DEFAULT_NEMO_CURATOR_METRICS_PATH  # noqa: S108


class TestPrometheusYamlTemplate:
    def test_template_is_valid_yaml(self) -> None:
        """PROMETHEUS_YAML_TEMPLATE should parse as valid YAML."""
        config = yaml.safe_load(PROMETHEUS_YAML_TEMPLATE)
        assert config is not None
        assert "scrape_configs" in config
        assert "global" in config

    def test_template_has_empty_file_sd_configs(self) -> None:
        """PROMETHEUS_YAML_TEMPLATE should have an empty file_sd_configs list."""
        config = yaml.safe_load(PROMETHEUS_YAML_TEMPLATE)
        file_sd_configs = config["scrape_configs"][0]["file_sd_configs"]
        assert file_sd_configs == []

    def test_template_does_not_contain_hardcoded_ray_path(self) -> None:
        """PROMETHEUS_YAML_TEMPLATE should not contain /tmp/ray."""
        assert "/tmp/ray" not in PROMETHEUS_YAML_TEMPLATE  # noqa: S108


class TestGetPrometheusPort:
    def test_reads_port_from_file(self, tmp_path: pathlib.Path) -> None:
        """get_prometheus_port should read port from the port file."""
        port_file = tmp_path / PROMETHEUS_PORT_FILE
        port_file.write_text("9999")
        assert get_prometheus_port(str(tmp_path)) == 9999

    def test_returns_default_when_no_file(self, tmp_path: pathlib.Path) -> None:
        """get_prometheus_port should return 9090 when port file doesn't exist."""
        assert get_prometheus_port(str(tmp_path)) == 9090

    def test_returns_default_on_invalid_content(self, tmp_path: pathlib.Path) -> None:
        """get_prometheus_port should return 9090 when port file has invalid content."""
        port_file = tmp_path / PROMETHEUS_PORT_FILE
        port_file.write_text("not_a_number")
        assert get_prometheus_port(str(tmp_path)) == 9090


class TestIsProcessRunningFromPidfile:
    def test_returns_false_when_no_file(self, tmp_path: pathlib.Path) -> None:
        """Should return False when PID file doesn't exist."""
        pid_file = str(tmp_path / "nonexistent.pid")
        assert _is_process_running_from_pidfile(pid_file) is False

    def test_returns_false_on_invalid_pid_content(self, tmp_path: pathlib.Path) -> None:
        """Should return False when PID file has invalid content."""
        pid_file = tmp_path / "test.pid"
        pid_file.write_text("garbage")
        assert _is_process_running_from_pidfile(str(pid_file)) is False

    def test_returns_false_on_stale_pid(self, tmp_path: pathlib.Path) -> None:
        """Should return False when PID doesn't correspond to a running process."""
        pid_file = tmp_path / "test.pid"
        # Use a PID that's very unlikely to exist
        pid_file.write_text("999999999")
        assert _is_process_running_from_pidfile(str(pid_file)) is False

    def test_returns_true_for_current_process(self, tmp_path: pathlib.Path) -> None:
        """Should return True for the current running process."""
        pid_file = tmp_path / "test.pid"
        pid_file.write_text(str(os.getpid()))
        assert _is_process_running_from_pidfile(str(pid_file)) is True


class TestIsPrometheusRunning:
    def test_returns_false_when_no_pidfile(self, tmp_path: pathlib.Path) -> None:
        """Should return False when there's no PID file in the metrics dir."""
        assert is_prometheus_running(str(tmp_path)) is False

    def test_returns_false_with_stale_pid(self, tmp_path: pathlib.Path) -> None:
        """Should return False when PID file exists but process is dead."""
        pid_file = tmp_path / PROMETHEUS_PID_FILE
        pid_file.write_text("999999999")
        assert is_prometheus_running(str(tmp_path)) is False

    def test_returns_true_with_live_process(self, tmp_path: pathlib.Path) -> None:
        """Should return True when PID file points to a running process."""
        pid_file = tmp_path / PROMETHEUS_PID_FILE
        pid_file.write_text(str(os.getpid()))
        assert is_prometheus_running(str(tmp_path)) is True


class TestIsGrafanaRunning:
    def test_returns_false_when_no_pidfile(self, tmp_path: pathlib.Path) -> None:
        """Should return False when there's no PID file in the metrics dir."""
        assert is_grafana_running(str(tmp_path)) is False

    def test_returns_false_with_stale_pid(self, tmp_path: pathlib.Path) -> None:
        """Should return False when PID file exists but process is dead."""
        pid_file = tmp_path / GRAFANA_PID_FILE
        pid_file.write_text("999999999")
        assert is_grafana_running(str(tmp_path)) is False

    def test_returns_true_with_live_process(self, tmp_path: pathlib.Path) -> None:
        """Should return True when PID file points to a running process."""
        pid_file = tmp_path / GRAFANA_PID_FILE
        pid_file.write_text(str(os.getpid()))
        assert is_grafana_running(str(tmp_path)) is True


class TestAddRayPrometheusMetricsServiceDiscovery:
    def _write_empty_prometheus_config(self, metrics_dir: pathlib.Path) -> str:
        """Helper to write the empty prometheus config template."""
        config_path = os.path.join(str(metrics_dir), "prometheus.yml")
        with open(config_path, "w") as f:
            f.write(PROMETHEUS_YAML_TEMPLATE)
        return config_path

    @patch("nemo_curator.metrics.utils.requests.post")
    @patch("nemo_curator.metrics.utils.get_prometheus_port", return_value=9090)
    def test_adds_path_to_empty_config(self, mock_port: object, mock_post: object, tmp_path: pathlib.Path) -> None:
        """Should add discovery path to an empty file_sd_configs list."""
        self._write_empty_prometheus_config(tmp_path)
        (tmp_path / PROMETHEUS_PORT_FILE).write_text("9090")

        add_ray_prometheus_metrics_service_discovery("/custom/ray/temp", str(tmp_path))

        with open(tmp_path / "prometheus.yml") as f:
            config = yaml.safe_load(f)
        all_paths = _get_all_discovery_paths(config)
        assert "/custom/ray/temp/prom_metrics_service_discovery.json" in all_paths

    @patch("nemo_curator.metrics.utils.requests.post")
    @patch("nemo_curator.metrics.utils.get_prometheus_port", return_value=9090)
    def test_does_not_add_duplicate(self, mock_port: object, mock_post: object, tmp_path: pathlib.Path) -> None:
        """Should not add duplicate discovery path."""
        self._write_empty_prometheus_config(tmp_path)
        (tmp_path / PROMETHEUS_PORT_FILE).write_text("9090")

        add_ray_prometheus_metrics_service_discovery("/custom/ray/temp", str(tmp_path))
        add_ray_prometheus_metrics_service_discovery("/custom/ray/temp", str(tmp_path))

        with open(tmp_path / "prometheus.yml") as f:
            config = yaml.safe_load(f)
        all_paths = _get_all_discovery_paths(config)
        assert all_paths.count("/custom/ray/temp/prom_metrics_service_discovery.json") == 1

    @patch("nemo_curator.metrics.utils.requests.post")
    @patch("nemo_curator.metrics.utils.get_prometheus_port", return_value=9090)
    def test_adds_multiple_different_paths(
        self,
        mock_port: object,
        mock_post: object,
        tmp_path: pathlib.Path,
    ) -> None:
        """Should support multiple different Ray cluster paths."""
        self._write_empty_prometheus_config(tmp_path)
        (tmp_path / PROMETHEUS_PORT_FILE).write_text("9090")

        add_ray_prometheus_metrics_service_discovery("/ray/cluster1", str(tmp_path))
        add_ray_prometheus_metrics_service_discovery("/ray/cluster2", str(tmp_path))

        with open(tmp_path / "prometheus.yml") as f:
            config = yaml.safe_load(f)
        all_paths = _get_all_discovery_paths(config)
        assert "/ray/cluster1/prom_metrics_service_discovery.json" in all_paths
        assert "/ray/cluster2/prom_metrics_service_discovery.json" in all_paths


class TestRemoveRayPrometheusMetricsServiceDiscovery:
    def _write_config_with_paths(self, metrics_dir: pathlib.Path, paths: list[str]) -> str:
        """Helper to write prometheus config with specific discovery paths as separate entries."""
        config = yaml.safe_load(PROMETHEUS_YAML_TEMPLATE)
        config["scrape_configs"][0]["file_sd_configs"] = [{"files": [p]} for p in paths]
        config_path = os.path.join(str(metrics_dir), "prometheus.yml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return config_path

    def test_removes_correct_path(self, tmp_path: pathlib.Path) -> None:
        """Should remove only the specified Ray cluster's discovery path."""
        paths = [
            "/ray/cluster1/prom_metrics_service_discovery.json",
            "/ray/cluster2/prom_metrics_service_discovery.json",
        ]
        self._write_config_with_paths(tmp_path, paths)

        remove_ray_prometheus_metrics_service_discovery("/ray/cluster1", str(tmp_path))

        with open(tmp_path / "prometheus.yml") as f:
            config = yaml.safe_load(f)
        all_paths = _get_all_discovery_paths(config)
        assert "/ray/cluster1/prom_metrics_service_discovery.json" not in all_paths
        assert "/ray/cluster2/prom_metrics_service_discovery.json" in all_paths

    def test_noop_when_path_not_present(self, tmp_path: pathlib.Path) -> None:
        """Should be a no-op when the path isn't in the config."""
        paths = ["/ray/cluster1/prom_metrics_service_discovery.json"]
        self._write_config_with_paths(tmp_path, paths)

        remove_ray_prometheus_metrics_service_discovery("/ray/nonexistent", str(tmp_path))

        with open(tmp_path / "prometheus.yml") as f:
            config = yaml.safe_load(f)
        all_paths = _get_all_discovery_paths(config)
        assert all_paths == ["/ray/cluster1/prom_metrics_service_discovery.json"]

    def test_noop_when_no_config_file(self, tmp_path: pathlib.Path) -> None:
        """Should silently return when prometheus.yml doesn't exist."""
        remove_ray_prometheus_metrics_service_discovery("/ray/cluster1", str(tmp_path))

    def test_noop_when_empty_file_sd_configs(self, tmp_path: pathlib.Path) -> None:
        """Should silently return when file_sd_configs is empty."""
        config_path = os.path.join(str(tmp_path), "prometheus.yml")
        with open(config_path, "w") as f:
            f.write(PROMETHEUS_YAML_TEMPLATE)

        remove_ray_prometheus_metrics_service_discovery("/ray/cluster1", str(tmp_path))
