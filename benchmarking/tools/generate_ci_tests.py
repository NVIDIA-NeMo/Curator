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

DEFAULT_TIME = "00:10:00"


def generate_job(entry: dict) -> dict:
    """
    Generate a GitLab CI job for a single benchmark entry.

    Args:
        entry: Dictionary from nightly-benchmark.yaml entries list

    Returns:
        job: Dictionary defining the GitLab CI job
    """
    ci = entry["ci"]
    ray = entry.get("ray", {})
    job = {
        "extends": ".curator_benchmark_test",
        "stage": "benchmark",
        "variables": {
            "ENTRY_NAME": entry["name"],
            "TEST_LEVEL": ci["scope"],
            "TIME": DoubleQuotedScalarString(ci.get("time", DEFAULT_TIME)),
            "CPUS_PER_TASK": str(ray.get("num_cpus", "")),
        },
    }

    if ci.get("known_issue", False):
        job["allow_failure"] = True

    return job


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

    if scope == "NONE":
        scope = "nightly"

    pipeline = {
        "include": ["curator/curator_ci_template.yml"],
    }

    entries = config.get("entries", [])
    job_count = 0
    for entry in entries:
        ci = entry.get("ci")
        if ci is None:
            continue
        if ci.get("scope") != scope:
            continue
        if not entry.get("enabled", True):
            continue

        pipeline[entry["name"]] = generate_job(entry)
        job_count += 1

    if job_count == 0:
        msg = f"No benchmark entries found with ci.scope='{scope}' in {config_path}"
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
