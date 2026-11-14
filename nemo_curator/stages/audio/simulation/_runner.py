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

"""Validate resolved FastMSS settings and enrich its recording manifest.

Executed as a script in the simulation Python environment; keep Curator imports
out of this module so that only the engine's dependencies are required there.
"""

import argparse
import contextlib
import gzip
import json
import os
import re
import runpy
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

_MIN_SPEAKERS = 2


def validate_config(config: dict) -> None:
    """Reject settings the unmodified upstream engine cannot safely generate."""
    minimum, maximum = config.get("min_max_spk", [2, 4])
    if minimum < _MIN_SPEAKERS or maximum < minimum:
        msg = "FastMSS requires min_max_spk with 2 <= minimum <= maximum; single-speaker recipes are unsupported."
        raise ValueError(msg)
    if config.get("save_anechoic", False) and (
        not config.get("reverberate", False) or config.get("reverb_prob", 0.5) != 1
    ):
        msg = "FastMSS save_anechoic=true requires reverberate=true and reverb_prob=1; dry meetings are unsupported."
        raise ValueError(msg)


def enrich_manifest(output_dir: Path) -> None:
    """Join supervision text per recording, retaining overlap and other fields."""
    manifest = output_dir / "nemo_manifest.json"
    if not manifest.is_file() or not manifest.stat().st_size:
        return  # The parent reports missing/empty manifests with the child log path.
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if all(isinstance(row.get("text"), str) and row["text"].strip() for row in rows):
        return
    supervisions = defaultdict(list)
    for path in (output_dir / "manifests").glob("synth-*-train-supervisions.jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                supervision = json.loads(line)
                supervisions[supervision["recording_id"]].append(supervision)
    recordings = {}
    for path in (output_dir / "manifests").glob("synth-*-train-recordings.jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                recording = json.loads(line)
                recordings[recording["sources"][0]["source"]] = recording["id"]
    for row in rows:
        if isinstance(row.get("text"), str) and row["text"].strip():
            continue
        recording_id = recordings.get(row["audio_filepath"])
        segments = sorted(supervisions[recording_id], key=lambda segment: (segment["start"], segment["id"]))
        if not segments or any(not isinstance(s.get("text"), str) or not s["text"].strip() for s in segments):
            msg = f"FastMSS cannot build reference text: missing supervision transcripts for {row['audio_filepath']}"
            raise ValueError(msg)
        row["text"] = " ".join(segment["text"].strip() for segment in segments)
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(manifest)


def _record_config(output_dir: Path, config: dict, isolated: bool) -> None:
    """Persist the effective worker configuration beside isolated run artifacts."""
    if isolated and output_dir.is_dir():
        (output_dir / "effective_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _literal_values(value: object) -> object:
    """Keep resolved strings literal when Hydra reads the execution YAML again."""
    if isinstance(value, str):
        return re.sub(r"(\\*)\$\{", lambda match: "\\" * (2 * len(match[1]) + 1) + "${", value)
    if isinstance(value, list):
        return [_literal_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _literal_values(item) for key, item in value.items()}
    return value


def _execution_config(args: argparse.Namespace, overrides: list[str]) -> tuple[dict, Path]:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    if args.planned_config:
        if not args.isolate_work_dirs or overrides or args.jobs_per_config < 1:
            msg = "Planned configuration requires isolated paths, positive jobs and no additional overrides"
            raise ValueError(msg)
        planned = Path(args.planned_config).resolve()
        config = json.loads(planned.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            msg = "Planned configuration must be a resolved mapping"
            raise ValueError(msg)
        config.update(
            output_dir=str(planned.parent),
            n_jobs=args.jobs_per_config,
            stage=0,
            save_rttm=True,
            save_nemo_manifest=True,
        )
    else:
        with initialize_config_dir(config_dir=args.config_path, version_base=None):
            config = OmegaConf.to_container(compose(config_name=args.config_name, overrides=overrides), resolve=True)
    output_dir = Path(config["output_dir"])
    if args.isolate_work_dirs:
        config.update(data_dir=str(output_dir / "data"), rir_dir=str(output_dir / "rir"))
    if args.max_meetings != -1:
        if args.max_meetings < 1:
            msg = "max_meetings must be positive or -1"
            raise ValueError(msg)
        count = config.get("n_meetings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            msg = "n_meetings must be a positive integer"
            raise ValueError(msg)
        config["n_meetings"] = min(count, args.max_meetings)
    for key, expected in (("n_meetings", args.expected_meetings), ("duration", args.expected_duration)):
        if expected is not None and config.get(key) != expected:
            msg = f"Native {key} changed since request planning; prepare the request again"
            raise ValueError(msg)
    validate_config(config)
    return config, output_dir


def _frozen_arguments(
    config: dict, directory: str, overrides: list[str], *, config_name: str = "meeting"
) -> list[str]:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    path = Path(directory) / f"{config_name}.yaml"
    OmegaConf.save(OmegaConf.create(_literal_values(config)), path)
    if "output_dir" in config:
        overrides = [f"output_dir={json.dumps(_literal_values(config['output_dir']))}", *overrides]
    with initialize_config_dir(config_dir=directory, version_base=None):
        actual = OmegaConf.to_container(compose(config_name=config_name, overrides=overrides), resolve=True)
    if actual != config:
        msg = "Frozen native configuration changed during serialization"
        raise ValueError(msg)
    return [f"--config-path={directory}", f"--config-name={config_name}", *overrides]


def main() -> None:
    """Preflight the effective Hydra config before running any engine stages."""
    parser = argparse.ArgumentParser()
    parser.add_argument("script")
    parser.add_argument("--installed-fastmss", action="store_true")
    parser.add_argument("--isolate-work-dirs", action="store_true")
    parser.add_argument("--max-meetings", type=int, default=-1)
    parser.add_argument("--expected-meetings", type=int)
    parser.add_argument("--expected-duration", type=float)
    parser.add_argument("--planned-config")
    parser.add_argument("--jobs-per-config", type=int, default=1)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--config-name", required=True)
    args, overrides = parser.parse_known_args()
    if args.installed_fastmss:
        from importlib.metadata import PackageNotFoundError, distribution

        try:
            package = distribution("fastmss")
        except PackageNotFoundError as error:
            msg = "FastMSS is not installed in the worker Python; install nemo-curator[audio_simulation]."
            raise RuntimeError(msg) from error
        if args.script not in {str(path) for path in package.files or []}:
            msg = f"Installed FastMSS does not provide the recipe: {args.script}"
            raise FileNotFoundError(msg)
        args.script = str(package.locate_file(args.script))
        os.environ["FASTMSS_ROOT"] = str(package.locate_file(""))
    resolved, output_dir = _execution_config(args, overrides)
    # Match direct script execution; the sibling Curator fastmss.py would shadow
    # the upstream fastmss package if this runner's directory stayed first.
    sys.path[0] = str(Path(args.script).parent)
    freeze = bool(args.planned_config or args.isolate_work_dirs or args.max_meetings != -1)
    if freeze:
        output_dir.mkdir(parents=True, exist_ok=True)
    context = (
        tempfile.TemporaryDirectory(prefix="fastmss-config-", dir=output_dir)
        if freeze
        else contextlib.nullcontext(None)
    )
    with context as directory:
        if freeze:
            hydra_overrides = [item for item in overrides if item.lstrip("+").startswith(("hydra.", "hydra/"))]
            if args.isolate_work_dirs:
                hydra_overrides.extend(
                    [
                        "hydra.mode=RUN",
                        "hydra.output_subdir=null",
                        "hydra/job_logging=disabled",
                        "hydra/hydra_logging=disabled",
                        f"hydra.run.dir={json.dumps(str(output_dir))}",
                        "hydra.job.chdir=false",
                    ]
                )
            native_arguments = _frozen_arguments(resolved, directory, hydra_overrides, config_name=args.config_name)
        else:
            native_arguments = [f"--config-path={args.config_path}", f"--config-name={args.config_name}", *overrides]
        sys.argv = [args.script, *native_arguments]
        runpy.run_path(args.script, run_name="__main__")
    _record_config(output_dir, resolved, args.isolate_work_dirs)
    enrich_manifest(output_dir)


if __name__ == "__main__":
    main()
