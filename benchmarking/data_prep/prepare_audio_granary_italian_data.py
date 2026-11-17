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

"""Stage open Italian FLEURS recordings as an input manifest for the Granary v2 European pipelines."""

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
DEFAULT_HF_CONFIG = "it_it"
DEFAULT_HF_SPLIT = "test"
DEFAULT_CACHE_DIR = "/tmp/curator/fleurs_cache"  # noqa: S108
SAMPLE_RATE = 16_000  # FLEURS audio is 16 kHz mono; the TSV counts samples


def stage_dataset(  # noqa: PLR0913
    output_path: Path,
    cache_dir: str = DEFAULT_CACHE_DIR,
    hf_repo_id: str = DEFAULT_HF_REPO_ID,
    hf_revision: str = DEFAULT_HF_REVISION,
    hf_config: str = DEFAULT_HF_CONFIG,
    hf_split: str = DEFAULT_HF_SPLIT,
) -> None:
    output_path = output_path.resolve()
    manifest_path = output_path / "manifest.jsonl"
    if manifest_path.is_file():
        logger.info(f"Reusing staged FLEURS manifest at {manifest_path}")
        return

    def download(name: str) -> str:
        return hf_hub_download(
            hf_repo_id, f"data/{hf_config}/{name}", repo_type="dataset", revision=hf_revision, cache_dir=cache_dir
        )

    tsv_path = download(f"{hf_split}.tsv")
    archive_path = download(f"audio/{hf_split}.tar.gz")

    audio_dir = output_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as archive:
        # Extract by member name only, so a crafted path in the archive cannot leave audio_dir.
        for member in archive:
            if member.isfile() and member.name.endswith(".wav"):
                (audio_dir / Path(member.name).name).write_bytes(archive.extractfile(member).read())

    temporary_manifest = manifest_path.with_suffix(".jsonl.tmp")
    clips = 0
    duration_s = 0.0
    try:
        with temporary_manifest.open("w", encoding="utf-8") as manifest_file, open(tsv_path, encoding="utf-8") as tsv:
            # Columns: id, file_name, raw transcription, transcription, characters, num_samples, gender.
            for utterance_id, file_name, _, transcription, _, num_samples, _ in csv.reader(
                tsv, delimiter="\t", quoting=csv.QUOTE_NONE
            ):
                audio_path = audio_dir / file_name
                if not audio_path.is_file():
                    continue  # the TSV lists a few utterances the split's archive does not contain
                row = {
                    "audio_item_id": f"{hf_config}-{hf_split}-{utterance_id}-{audio_path.stem}",
                    "audio_filepath": str(audio_path),
                    "duration": int(num_samples) / SAMPLE_RATE,
                    "text": transcription,
                }
                manifest_file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                clips += 1
                duration_s += row["duration"]
        if not clips:
            msg = f"{hf_repo_id}/{hf_config}/{hf_split} produced no clips"
            raise RuntimeError(msg)  # noqa: TRY301
        temporary_manifest.replace(manifest_path)
    except Exception:
        temporary_manifest.unlink(missing_ok=True)
        raise

    logger.success(f"Staged {clips} recordings / {duration_s / 3600:.4f} hours at {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--hf-repo-id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--hf-revision", default=DEFAULT_HF_REVISION)
    parser.add_argument("--hf-config", default=DEFAULT_HF_CONFIG)
    parser.add_argument("--hf-split", default=DEFAULT_HF_SPLIT)
    stage_dataset(**vars(parser.parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
