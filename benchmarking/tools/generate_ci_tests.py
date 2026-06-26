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

import argparse
import datetime as dt
import os
from collections.abc import Mapping
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

yaml = YAML()
yaml.default_flow_style = False
yaml.preserve_quotes = True

# Fallbacks used only when nightly-benchmark.yaml omits the corresponding key; the YAML config
# is the source of truth for these timeout knobs.
DEFAULT_TIMEOUT_S = 7200  # mirrors Session.default_timeout_s in runner/session.py
DEFAULT_CLEANUP_TIMEOUT_S = 60
DEFAULT_MIN_TIMEOUT_S = 600


def seconds_to_time(seconds: int) -> str:
    """
    Convert integer seconds to HH:MM:SS format.

    Args:
        seconds: Number of seconds to convert

    Returns:
        time_str: Formatted time string (e.g. 1000 -> "00:16:40")
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _pipeline_timestamp(created_at: str | None) -> str:
    """Return a path-safe UTC timestamp for generated nightly session names."""
    if created_at:
        try:
            parsed = dt.datetime.fromisoformat(created_at)
            return parsed.astimezone(dt.UTC).strftime("%Y_%m_%d__%H_%M_%S_UTC")
        except ValueError:
            pass
    return dt.datetime.now(dt.UTC).strftime("%Y_%m_%d__%H_%M_%S_UTC")


def session_name_from_env(env: Mapping[str, str] = os.environ) -> str | None:
    """Return the generated-pipeline session name, if this run should define one."""
    explicit = env.get("NEMO_CI_SESSION_NAME", "").strip()
    if explicit:
        return explicit

    if env.get("CI_PIPELINE_SOURCE") == "schedule" or env.get("PARENT_PIPELINE_SOURCE") == "schedule":
        return f"nightly-{_pipeline_timestamp(env.get('CI_PIPELINE_CREATED_AT'))}"

    return None


def generate_job(entry: dict, scope: str, default_timeout_s: int, cleanup_timeout_s: int, min_timeout_s: int) -> dict:
    """
    Generate a GitLab CI job for a single benchmark entry.

    Args:
        entry: Dictionary from nightly-benchmark.yaml entries list
        scope: CI scope (e.g. "nightly")
        default_timeout_s: Timeout used for entries that omit "timeout_s"
        cleanup_timeout_s: Buffer added on top of every entry's timeout for post-run cleanup
        min_timeout_s: Floor on the generated job time to cover container setup overhead

    Returns:
        job: Dictionary defining the GitLab CI job
    """
    ray = entry.get("ray", {})
    # SLURM wall-clock = entry's effective timeout + a fixed cleanup buffer, floored at
    # min_timeout_s so short entries get enough time for container setup before their run starts.
    timeout_s = max(entry.get("timeout_s", default_timeout_s) + cleanup_timeout_s, min_timeout_s)
    time_str = seconds_to_time(timeout_s)

    return {
        "extends": ".curator_benchmark_test",
        "stage": "benchmark",
        "variables": {
            "ENTRY_NAME": entry["name"],
            "TEST_LEVEL": scope,
            "TIME": DoubleQuotedScalarString(time_str),
            "CPUS_PER_TASK": str(ray.get("num_cpus", "")),
        },
    }


def generate_pipeline(curator_dir: str, scope: str, session_name: str | None = None) -> dict:
    """
    Generate a GitLab CI pipeline from Curator benchmark entries.

    Args:
        curator_dir: Path to the Curator repository
        scope: Scope of the testing (nightly, release, test)
        session_name: Optional session name to set for all generated benchmark jobs

    Returns:
        pipeline: Dictionary defining the GitLab CI pipeline
    """
    config_path = Path(curator_dir) / "benchmarking" / "nightly-benchmark.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.load(f)

    if scope == "NONE":
        scope = "nightly"

    default_timeout_s = config.get("default_timeout_s", DEFAULT_TIMEOUT_S)
    cleanup_timeout_s = config.get("cleanup_timeout_s", DEFAULT_CLEANUP_TIMEOUT_S)
    min_timeout_s = config.get("min_timeout_s", DEFAULT_MIN_TIMEOUT_S)

    pipeline = {
        "include": ["curator/curator_ci_template.yml"],
    }
    if session_name:
        pipeline["variables"] = {"NEMO_CI_SESSION_NAME": session_name}

    entries = config.get("entries", [])
    job_count = 0
    for entry in entries:
        if not entry.get("enabled", True):
            continue

        pipeline[entry["name"]] = generate_job(entry, scope, default_timeout_s, cleanup_timeout_s, min_timeout_s)
        job_count += 1

    if job_count == 0:
        msg = f"No benchmark entries found for scope='{scope}' in {config_path}"
        raise ValueError(msg)

    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GitLab CI jobs for Curator benchmarks")
    parser.add_argument(
        "--curator-dir",
        type=str,
        required=True,
        help="Path to Curator directory",
    )
    parser.add_argument(
        "--scope",
        type=str,
        required=True,
        help="Scope of the tests (nightly, release, test)",
    )

    args = parser.parse_args()

    session_name = session_name_from_env()
    pipeline = generate_pipeline(args.curator_dir, args.scope, session_name=session_name)

    output_file = "generated_curator_benchmark_tests.yml"
    with open(output_file, "w") as f:
        yaml.dump(pipeline, f)

    job_count = len([k for k in pipeline if k not in {"include", "variables"}])
    print(f"Generated pipeline with {job_count} jobs -> {output_file}")
    if session_name:
        print(f"Using benchmark session name: {session_name}")


if __name__ == "__main__":
    main()
