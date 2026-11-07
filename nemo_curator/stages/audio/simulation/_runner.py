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
import gzip
import json
import runpy
import sys
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


def main() -> None:
    """Preflight the effective Hydra config before running any engine stages."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("script")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--config-name", required=True)
    args, overrides = parser.parse_known_args()
    with initialize_config_dir(config_dir=args.config_path, version_base=None):
        config = compose(config_name=args.config_name, overrides=overrides)
        validate_config(OmegaConf.to_container(config, resolve=True))
        output_dir = Path(config.output_dir)
    sys.argv = [args.script, *sys.argv[2:]]
    # Match direct script execution; the sibling Curator fastmss.py would shadow
    # the upstream fastmss package if this runner's directory stayed first.
    sys.path[0] = str(Path(args.script).parent)
    runpy.run_path(args.script, run_name="__main__")
    enrich_manifest(output_dir)


if __name__ == "__main__":
    main()
