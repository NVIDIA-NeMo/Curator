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

"""Stage open Indic FLEURS recordings (CC-BY-4.0) as a Granary v2 Indic pipeline input of a chosen duration.

The manifest rows are ``audio_item_id``, ``audio_filepath`` and ``duration``; the recording transcript is not written,
because every VAD clip would inherit it as if it were a clip label."""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from loguru import logger

DEFAULT_HF_REPO_ID = "google/fleurs"  # CC-BY-4.0, not gated
DEFAULT_HF_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"  # pragma: allowlist secret
DEFAULT_LANGUAGES = "hi_in,bn_in,ta_in,te_in,mr_in,gu_in,kn_in,ml_in,pa_in"
DEFAULT_CACHE_DIR = "/tmp/curator/fleurs_cache"  # noqa: S108
SAMPLE_RATE = 16_000  # FLEURS audio is 16 kHz mono; the TSV counts samples


def _language_rows(config: str, cache_dir: str, hf_repo_id: str, hf_revision: str) -> list[tuple[str, str, float]]:
    """(utterance id, file name, seconds) of the config's train split, in TSV order."""
    tsv_path = hf_hub_download(
        hf_repo_id, f"data/{config}/train.tsv", repo_type="dataset", revision=hf_revision, cache_dir=cache_dir
    )
    with open(tsv_path, encoding="utf-8") as tsv:
        # Columns: id, file_name, raw transcription, transcription, characters, num_samples, gender.
        return [
            (utt, name, int(samples) / SAMPLE_RATE)
            for utt, name, _, _, _, samples, _ in csv.reader(tsv, delimiter="\t", quoting=csv.QUOTE_NONE)
        ]


def stage_dataset(  # noqa: PLR0913
    output_path: Path,
    target_audio_hours: float,
    languages: str = DEFAULT_LANGUAGES,
    cache_dir: str = DEFAULT_CACHE_DIR,
    hf_repo_id: str = DEFAULT_HF_REPO_ID,
    hf_revision: str = DEFAULT_HF_REVISION,
) -> None:
    output_path = output_path.resolve()
    manifest_path = output_path / "manifest.jsonl"
    if manifest_path.is_file():
        logger.info(f"Reusing staged manifest at {manifest_path}")
        return
    configs = languages.split(",")
    per_language_s = target_audio_hours * 3600 / len(configs)  # an equal share of the hours per language
    audio_dir = output_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(".jsonl.tmp")
    clips = 0
    seconds = 0.0
    try:
        with temporary_manifest.open("w", encoding="utf-8") as manifest_file:
            for config in configs:
                chosen, taken, written = {}, 0.0, 0.0
                for utt, name, duration in _language_rows(config, cache_dir, hf_repo_id, hf_revision):
                    if taken >= per_language_s:
                        break
                    chosen[name] = (utt, duration)
                    taken += duration
                archive = hf_hub_download(
                    hf_repo_id,
                    f"data/{config}/audio/train.tar.gz",
                    repo_type="dataset",
                    revision=hf_revision,
                    cache_dir=cache_dir,
                )
                with tarfile.open(archive) as tar:
                    for member in tar:
                        file_name = Path(member.name).name
                        if not member.isfile() or file_name not in chosen:
                            continue
                        utt, duration = chosen.pop(file_name)
                        audio_path = audio_dir / f"{config}-{file_name}"
                        audio_path.write_bytes(tar.extractfile(member).read())
                        record = {
                            "audio_item_id": f"{config}-train-{utt}-{Path(file_name).stem}",
                            "audio_filepath": str(audio_path),
                            "duration": duration,
                        }
                        manifest_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                        clips += 1
                        written += duration
                        seconds += duration
                # Count the audio actually written: a transcript row without an archive member adds nothing.
                if written < per_language_s:
                    msg = (
                        f"{config} train supplies {written / 3600:.2f} of the {per_language_s / 3600:.2f} hours needed "
                        f"({len(chosen)} selected rows have no audio in the archive)"
                    )
                    raise RuntimeError(msg)  # noqa: TRY301
        temporary_manifest.replace(manifest_path)
    except Exception:
        temporary_manifest.unlink(missing_ok=True)
        raise
    logger.success(
        f"Staged {clips} recordings / {seconds / 3600:.3f} hours in {len(configs)} languages at {manifest_path}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--target-audio-hours", type=float, required=True)
    parser.add_argument("--languages", default=DEFAULT_LANGUAGES, help="comma-separated FLEURS configs")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--hf-repo-id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--hf-revision", default=DEFAULT_HF_REVISION)
    stage_dataset(**vars(parser.parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
