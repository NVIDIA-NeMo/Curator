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
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

yaml = YAML()
yaml.default_flow_style = False
yaml.preserve_quotes = True

MIN_TIME_S = 600


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



def generate_job(entry: dict, scope: str) -> dict:
    """
    Generate a GitLab CI job for a single benchmark entry.

    Args:
        entry: Dictionary from nightly-benchmark.yaml entries list
        scope: CI scope (e.g. "nightly")

    Returns:
        job: Dictionary defining the GitLab CI job
    """
    ray = entry.get("ray", {})
    # Enforce a minimum of 10 minutes to allow for setup overhead
    timeout_s = max(entry.get("timeout_s", MIN_TIME_S), MIN_TIME_S)
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


def load_entry_overrides(curator_dir: str) -> dict:
    """Load per-entry field overrides from ci-h100-overrides.yaml if it exists."""
    overrides_path = Path(curator_dir) / "benchmarking" / "ci-h100-overrides.yaml"
    if not overrides_path.exists():
        return {}
    with open(overrides_path, encoding="utf-8") as f:
        data = yaml.load(f)
    return data.get("entry_overrides", {}) if data else {}


def generate_pipeline(curator_dir: str, scope: str) -> dict:
    """
    Generate a GitLab CI pipeline from Curator benchmark entries.

    Args:
        curator_dir: Path to the Curator repository
        scope: Scope of the testing (nightly, release, test)

    Returns:
        pipeline: Dictionary defining the GitLab CI pipeline
    """
    config_path = Path(curator_dir) / "benchmarking" / "nightly-benchmark.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.load(f)

    entry_overrides = load_entry_overrides(curator_dir)

    if scope == "NONE":
        scope = "nightly"

    pipeline = {
        "include": ["curator/curator_ci_template.yml"],
    }

    entries = config.get("entries", [])
    for entry in entries:
        overrides = entry_overrides.get(entry["name"])
        if overrides:
            entry.update(overrides)

    job_count = 0
    for entry in entries:
        if not entry.get("enabled", True):
            continue

        pipeline[entry["name"]] = generate_job(entry, scope)
        job_count += 1

    if job_count == 0:
        msg = f"No benchmark entries found for scope='{scope}' in {config_path}"
        raise ValueError(msg)

    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GitLab CI jobs for Curator benchmarks"
    )
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

    pipeline = generate_pipeline(args.curator_dir, args.scope)

    output_file = "generated_curator_benchmark_tests.yml"
    with open(output_file, "w") as f:
        yaml.dump(pipeline, f)

    job_count = len([k for k in pipeline if k != "include"])
    print(f"Generated pipeline with {job_count} jobs -> {output_file}")


if __name__ == "__main__":
    main()
