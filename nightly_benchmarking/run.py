"""Benchmark driver for ray-curator.

This tool executes a user-specified matrix of benchmark commands with strict
per-run Ray isolation, per-entry timeouts, environment capture, and pluggable
metric sinks (MLflow by default; W&B optional in the future).

Key properties:
- Matrix owns commands and parameters; the driver never mutates script flags.
- Each entry runs in isolation: start fresh Ray head, run command, stop Ray.
- Dataset path substitution via placeholders: {dataset:<name>,<format>}.
- Results are normalized to JSONL; artifacts include logs and env snapshots.
"""

import argparse
import json
import os
import pickle
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from nemo_curator.tasks import Task
from nemo_curator.tasks.utils import TaskPerfUtils
from nightly_benchmarking.sinks.factory import build_sink
from nightly_benchmarking.utils.datasets import DatasetResolver
from nightly_benchmarking.utils.env_capture import capture_environment_artifacts, collect_basic_env

# Local imports
from nightly_benchmarking.utils.matrix import MatrixConfig, MatrixEntry
from nightly_benchmarking.utils.process import run_command_with_timeout
from nightly_benchmarking.utils.ray_cluster import (
    start_ray_head,
    stop_ray_head,
    verify_ray_responsive,
)


def aggregate_task_metrics(tasks: list[Task], prefix: str | None = None) -> dict[str, Any]:
    metrics = {}
    tasks_metrics = TaskPerfUtils.collect_stage_metrics(tasks)
    # For each of the metric compute mean/std/sum and flatten the dict
    for stage_name, stage_data in tasks_metrics.items():
        for metric_name, values in stage_data.items():
            for agg_name, agg_func in [("sum", np.sum), ("mean", np.mean), ("std", np.std)]:
                stage_key = stage_name if prefix is None else f"{prefix}_{stage_name}"
                if len(values) > 0:
                    metrics[f"{stage_key}_{metric_name}_{agg_name}"] = float(agg_func(values))
                else:
                    metrics[f"{stage_key}_{metric_name}_{agg_name}"] = 0.0
    return metrics


def get_script_params_metrics(benchmark_results_dir: Path) -> dict[str, Any]:
    with open(benchmark_results_dir / "params.json") as f:
        script_params = json.load(f)
    with open(benchmark_results_dir / "metrics.json") as f:
        script_metrics = json.load(f)

    with open(benchmark_results_dir / "tasks.pkl", "rb") as f:
        script_tasks = pickle.load(f)  # noqa: S301
        if isinstance(script_tasks, list):
            script_metrics.update(aggregate_task_metrics(script_tasks, prefix="task"))
        elif isinstance(script_tasks, dict):
            for pipeline_name, pipeline_tasks in script_tasks.items():
                script_metrics.update(aggregate_task_metrics(pipeline_tasks, prefix=pipeline_name.lower()))

    return {"params": script_params, "metrics": script_metrics}


def create_dir(path: str) -> None:
    """Ensure parent directory exists without removing it."""
    Path(path).mkdir(parents=True, exist_ok=True)


def create_or_overwrite_dir(path: str) -> None:
    """Create directory, removing it if it exists."""
    if Path(path).exists():
        shutil.rmtree(path, ignore_errors=True)
    Path(path).mkdir(parents=True, exist_ok=True)


def run_entry(  # noqa: PLR0915
    entry: MatrixEntry,
    resolver: DatasetResolver,
    session_dir: str,
    default_timeout_s: int,
    delete_scratch_default: bool,
) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    started_at = time.time()

    # Build command from either explicit cmd or script+args under scripts dir
    cmd = entry.get_command_to_run()
    # Prepare entry directories first so we can substitute paths
    entry_dir = Path(session_dir) / entry.name
    # scratch_dir : This is the directory user can use to store scratch data; it'll be cleaned up after the entry is done
    # ray_cluster_dir : This is the directory where Ray cluster is started; it'll be cleaned up after the entry is done
    # logs_dir : This is the directory where logs are stored
    # run_artifacts_dir : This is the directory where artifacts are stored
    # benchmark_results_dir : This is the directory where benchmark results are stored
    scratch_dir, ray_cluster_dir, logs_dir, run_artifacts_dir, benchmark_results_dir = [
        (entry_dir / d).absolute() for d in ["scratch", "ray_cluster", "logs", "artifacts", "benchmark_results"]
    ]

    # Now substitute template placeholders with actual paths
    cmd = entry.substitute_datasets_in_cmd(cmd, resolver)
    cmd = entry.substitute_template_placeholders(cmd, str(entry_dir))

    # Clean up any existing RAY_ADDRESS from previous runs
    if "RAY_ADDRESS" in os.environ:
        logger.debug("Cleaning up existing RAY_ADDRESS from previous run")
        del os.environ["RAY_ADDRESS"]

    ray_client = None
    short_temp_dir = None

    # Determine whether to delete scratch for this entry (entry overrides session default)
    should_delete_scratch = (
        entry.delete_scratch if getattr(entry, "delete_scratch", None) is not None else delete_scratch_default
    )

    try:
        # Create directories individually
        for directory in [scratch_dir, ray_cluster_dir, logs_dir, run_artifacts_dir, benchmark_results_dir]:
            create_or_overwrite_dir(str(directory))

        # Capture environment artifacts AFTER creating directories
        capture_environment_artifacts(str(run_artifacts_dir))

        # Create log files for Ray startup
        ray_startup_log = str(logs_dir / "ray_startup.log")

        # Start Ray cluster
        ray_client, short_temp_dir = start_ray_head(
            num_cpus=int(entry.ray.get("num_cpus", os.cpu_count() or 1)),
            num_gpus=int(entry.ray.get("num_gpus", 0)),
            enable_object_spilling=bool(entry.ray.get("enable_object_spilling", False)),
            ray_log_file=ray_startup_log,
        )
        verify_ray_responsive(ray_client)

        # Build environment and command
        env = os.environ.copy()
        ray_address = f"localhost:{ray_client.ray_port}"
        env["RAY_ADDRESS"] = ray_address
        # Also set globally so Ray Data executor can find it
        os.environ["RAY_ADDRESS"] = ray_address
        logger.debug(f"Set RAY_ADDRESS={ray_address}")
        # Timeout
        timeout_s = int(entry.timeout_s or default_timeout_s)

        # Execute command with timeout
        logger.info(f"\t\tRunning command {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        started_exec = time.time()
        completed = run_command_with_timeout(
            command=cmd,
            timeout_seconds=timeout_s,
            stdout_path=str(logs_dir / "stdout.log"),
            stderr_path=str(logs_dir / "stderr.log"),
            env=env,
        )
        ended_exec = time.time()

        # Show result
        duration = ended_exec - started_exec
        if completed["returncode"] == 0:
            logger.success(f"\t\t✅ Run Succeeded in {duration:.1f} seconds")
        else:
            logger.error(f"\t\t❌ Run Failed in {duration:.1f} seconds")
            if completed["timed_out"]:
                logger.warning(f"\t\t⏰ Timed out after {timeout_s}s")
        logger.info(f"\t\tLogs found in {logs_dir}")

        # Prepare normalized result
        basic_env = collect_basic_env()

        # Params
        ray_params = {
            "address": env["RAY_ADDRESS"],
            "num_cpus": ray_client.num_cpus,
            "num_gpus": ray_client.num_gpus,
            "enable_object_spilling": ray_client.enable_object_spilling,
        }
        run_params = {
            "name": entry.name,
            "cmd": cmd,
            "started_at": started_at,
            "ended_at": time.time(),
            "exec_started_at": started_exec,
            "logs_dir": str(logs_dir),
        }

        # Metrics
        run_metrics = {
            "exec_time_s": ended_exec - started_exec,
            "exit_code": completed["returncode"],
            "timed_out": completed["timed_out"],
        }

        # Script Params metrics
        script_params_metrics = get_script_params_metrics(benchmark_results_dir)

        result: dict[str, Any] = {
            "run": run_params,
            "ray": ray_params,
            "env": basic_env,
            "script_params_metrics": script_params_metrics,
        }

        Path(entry_dir / "results.json").write_text(json.dumps(result))

        # Determine run success status
        run_success = completed["returncode"] == 0

        return (
            result,
            run_success,
            {
                "params": {
                    **{f"run_{k}": v for k, v in run_params.items()},
                    **{f"ray_{k}": v for k, v in ray_params.items()},
                    **{f"env_{k}": v for k, v in basic_env.items()},
                    **{f"script_{k}": v for k, v in script_params_metrics["params"].items()},
                },
                "metrics": {
                    **{f"run_{k}": v for k, v in run_metrics.items()},
                    **{f"script_{k}": v for k, v in script_params_metrics["metrics"].items()},
                },
                "artifacts": [
                    str(run_artifacts_dir / "pip-freeze.txt"),
                    str(run_artifacts_dir / "conda-explicit.txt"),
                ],
            },
        )
    finally:
        if ray_client is not None:
            stop_ray_head(ray_client, short_temp_dir, str(ray_cluster_dir))

            # Clean up RAY_ADDRESS environment variable immediately after stopping cluster
            if "RAY_ADDRESS" in os.environ:
                logger.debug(f"Cleaning up RAY_ADDRESS={os.environ['RAY_ADDRESS']}")
                del os.environ["RAY_ADDRESS"]

        # Clean up the scratch dir if configured to delete
        if should_delete_scratch:
            shutil.rmtree(str(scratch_dir), ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ray-curator benchmark matrix")
    parser.add_argument("--matrix", required=True, help="Path to YAML matrix config")
    parser.add_argument("--datasets", required=True, help="Path to dataset_paths.json")
    parser.add_argument("--sink", default="none", choices=["mlflow", "wandb", "none"], help="Metrics sink kind")
    parser.add_argument(
        "--session-name", default=None, help="Optional human-readable session name (default nightly-run-<ts>)"
    )
    args = parser.parse_args()

    # Clean up any existing RAY_ADDRESS from previous sessions
    if "RAY_ADDRESS" in os.environ:
        logger.info(f"Cleaning up stale RAY_ADDRESS={os.environ['RAY_ADDRESS']} from previous session")
        del os.environ["RAY_ADDRESS"]

    cfg = MatrixConfig.load_yaml(args.matrix)
    create_dir(cfg.results_dir)  # Don't delete the parent results directory

    # Create session folder under results_dir
    session_name_prefix = args.session_name or "nightly-run"
    session_name = time.strftime(f"{session_name_prefix}__%Y-%m-%d__%H-%M-%S")
    session_dir = str((Path(cfg.results_dir) / session_name).absolute())
    create_dir(session_dir)  # Create session dir without deleting if exists

    resolver = DatasetResolver(args.datasets)

    # Create session-aware sink with configuration
    sink = build_sink(kind=args.sink, config={"mlflow": cfg.mlflow, "wandb": cfg.wandb})
    session_overall_success = True

    # Start session (parent run)
    logger.info(f"Started session {session_name}...")
    sink.start_session(session_name)

    try:
        results: list[dict[str, Any]] = []
        for entry in cfg.entries:
            logger.info(f"\tRunning {entry.name}")
            # Start run for this entry
            sink.start_run(run_name=entry.name)
            run_success = False  # Default to failed, will be updated if successful

            try:
                result, run_success, sink_data = run_entry(
                    entry=entry,
                    resolver=resolver,
                    session_dir=session_dir,
                    default_timeout_s=cfg.default_timeout_s,
                    delete_scratch_default=cfg.delete_scratch,
                )

                # Log to sink
                sink.log_params(sink_data["params"])
                sink.log_metrics(sink_data["metrics"])
                for artifact_path in sink_data["artifacts"]:
                    sink.log_artifact(artifact_path)

                results.append(result)
            except Exception as e:  # noqa: BLE001
                # Handle exceptions at bench_driver level
                run_success = False
                session_overall_success = False

                # Get the full traceback for better debugging
                error_traceback = traceback.format_exc()
                logger.error(f"\t\t❌ Entry failed with exception: {e}")
                logger.debug(f"Full traceback:\n{error_traceback}")

                result = {
                    "name": entry.name,
                    "run_id": f"{entry.name}-{int(time.time())}",
                    "error": str(e),
                    "traceback": error_traceback,
                    "success": False,
                }
                results.append(result)
            finally:
                # Always end the run, even if there was an exception
                sink.end_run(success=run_success)
                if not run_success:
                    session_overall_success = False

        # TODO: In future we can post to slack / upload to google drive
    finally:
        # Always end the session, even if there was an exception
        sink.end_session(success=session_overall_success)


if __name__ == "__main__":
    main()
