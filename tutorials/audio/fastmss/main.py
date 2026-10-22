# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Generate meeting mixtures with FastMSS and ingest them as Curator AudioTasks."""

import argparse
import json
from pathlib import Path

from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.simulation import FastMSS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fastmss-root", required=True, help="External FastMSS checkout on every worker")
    parser.add_argument("--config", required=True, action="append", help="FastMSS YAML; repeat for parallel shards")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--script-path", default="recipes/sim.py")
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--jobs-per-config", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--override", action="append", default=[], help="Hydra override, e.g. duration=60")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    # Keep the example's consolidated output separate from existing datasets.
    output_dir.mkdir(parents=True, exist_ok=False)
    kwargs = {} if args.python_executable is None else {"python_executable": args.python_executable}
    stage = FastMSS(
        config_paths=[str(Path(config).expanduser().resolve()) for config in args.config],
        fastmss_root=str(Path(args.fastmss_root).expanduser().resolve()),
        output_dir=str(output_dir),
        script_path=args.script_path,
        jobs_per_config=args.jobs_per_config,
        overrides=args.override,
        timeout_seconds=args.timeout_seconds,
        **kwargs,
    )
    pipeline = Pipeline(name="fastmss_training_data", stages=[stage])
    with RayClient():
        results = pipeline.run(XennaExecutor())
    if not results:
        msg = "FastMSS produced no meetings; check config paths and simulation logs"
        raise RuntimeError(msg)
    # Only publish the combined reference manifest after generation succeeds.
    manifest = output_dir / "meetings.jsonl"
    with manifest.open("x", encoding="utf-8") as output:
        for task in results or []:
            output.write(json.dumps(dict(task.data), ensure_ascii=False) + "\n")
    logger.info(f"Generated recording-level training manifest: {manifest}")


if __name__ == "__main__":
    main()
