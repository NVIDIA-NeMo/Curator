import os
import subprocess
import tempfile
import time

import requests

from ray_curator.core.client import get_ray_client

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

METRIC_NAME = "ray_gcs_actors_count"


def _query_prometheus(prometheus_port: int, metric_name: str = METRIC_NAME) -> list[dict]:
    """Query Prometheus for a specific metric and return the raw results list."""
    url = f"http://localhost:{prometheus_port}/api/v1/query"
    response = requests.get(url, params={"query": metric_name}, timeout=5)
    response.raise_for_status()
    payload = response.json()
    if payload["status"] != "success":  # pragma: no cover
        return []
    return payload["data"].get("result", [])


def _wait_for_metric(session_name: str, prometheus_port: int, timeout: int = 60) -> bool:
    """Poll Prometheus until the metric for *session_name* appears or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            results = _query_prometheus(prometheus_port)
        except requests.RequestException:
            # Prometheus may not be reachable yet, wait and retry.
            results = []
        for result in results:
            if result.get("metric", {}).get("SessionName") == session_name:
                return True
        time.sleep(2)
    return False


def _latest_session_name(ray_temp_dir: str, timeout: int = 30) -> str:
    """Return the latest Ray session directory name inside *ray_temp_dir*."""
    session_link = os.path.join(ray_temp_dir, "session_latest")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.islink(session_link):
            target = os.readlink(session_link)
            return os.path.basename(target.rstrip(os.sep))
        time.sleep(1)
    msg = "Ray session directory not created in time."
    raise RuntimeError(msg)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_single_cluster_metrics_visible():
    """Start one Ray cluster and ensure its metrics appear in Prometheus."""
    # Clear RAY_ADDRESS
    os.environ.pop("RAY_ADDRESS", None)
    with tempfile.TemporaryDirectory(prefix="ray_test_single_") as ray_tmp:
        client_info = get_ray_client(ray_temp_dir=ray_tmp)
        prometheus_port = client_info["prometheus_web_port"] or 9090  # Fallback if already running

        session_name = _latest_session_name(ray_tmp)
        assert _wait_for_metric(session_name, prometheus_port), (
            f"Metric for session {session_name} not found in Prometheus on port {prometheus_port}."
        )

    # TODO: This will kill all the ray processes. Find a better way to do this.
    # Cleanup : stop any Ray processes we started
    subprocess.run(["ray", "stop", "--force"], check=False, capture_output=True)  # noqa: S603, S607


def test_two_clusters_metrics_visible():
    """Start two separate Ray clusters and ensure Prometheus sees metrics for both."""
    # Clear RAY_ADDRESS
    os.environ.pop("RAY_ADDRESS", None)
    with (
        tempfile.TemporaryDirectory(prefix="ray_test_one_") as tmp1,
        tempfile.TemporaryDirectory(prefix="ray_test_two_") as tmp2,
    ):
        # First cluster
        info1 = get_ray_client(ray_temp_dir=tmp1)
        prometheus_port = info1["prometheus_web_port"] or 9090  # Prometheus port to query later
        session1 = _latest_session_name(tmp1)

        # Clear RAY_ADDRESS
        os.environ.pop("RAY_ADDRESS", None)
        # Second cluster : it should reuse the existing Prometheus instance
        # We need to wait to ensure Ray runs clients on the ports
        time.sleep(10)
        get_ray_client(ray_temp_dir=tmp2)
        session2 = _latest_session_name(tmp2)

        # Wait until metrics for both sessions appear
        assert _wait_for_metric(session1, prometheus_port), f"Metric for session {session1} not found in Prometheus."
        assert _wait_for_metric(session2, prometheus_port), f"Metric for session {session2} not found in Prometheus."

    # TODO: This will kill all the ray processes. Find a better way to do this.
    subprocess.run(["ray", "stop", "--force"], check=False, capture_output=True)  # noqa: S603, S607
