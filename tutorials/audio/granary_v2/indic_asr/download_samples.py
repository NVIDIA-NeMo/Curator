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

"""Download two public FLEURS Hindi clips and their reference transcripts."""

import argparse
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path
from urllib.request import Request, urlopen

import soundfile as sf

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"  # pragma: allowlist secret
BASE_URL = f"https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}/data/hi_in"
METADATA_SHA256 = "cea87c57a37a0d38ed0afce30e68a35ad7b3945414648430fee09a54ca7b72ba"  # pragma: allowlist secret
SAMPLES = {
    "10003386135529782184.wav": "eaf505bbf7f4ce5543af9242678b0a992f11d933a4878685b7bbfd817c4e422d",  # pragma: allowlist secret
    "10028896103651079035.wav": "f9910565cad24bb9a7a3129d1e6a171393fd041d339f961c8233ba0bd1276fef",  # pragma: allowlist secret
}
SAMPLE_RATE = 16000
ARCHIVE_PREFIX_BYTES = 2 * 1024 * 1024


def _download_prefix(url: str, limit: int) -> bytes:
    request = Request(url, headers={"Range": f"bytes=0-{limit - 1}"})  # noqa: S310
    with urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read(limit)


def _download_audio() -> dict[str, bytes]:
    # These two members fit in the first 2 MiB; never download the full archive.
    prefix = _download_prefix(f"{BASE_URL}/audio/dev.tar.gz", ARCHIVE_PREFIX_BYTES)
    payloads = {}
    with tarfile.open(fileobj=io.BytesIO(prefix), mode="r|gz") as archive:
        for member in archive:
            name = member.name.removeprefix("dev/")
            if member.name != f"dev/{name}" or name not in SAMPLES or not member.isfile():
                continue
            if member.size > ARCHIVE_PREFIX_BYTES:
                msg = f"Unexpected size for FLEURS sample: {name}"
                raise ValueError(msg)
            stream = archive.extractfile(member)
            if stream is None:
                msg = f"Missing FLEURS sample data: {name}"
                raise ValueError(msg)
            payload = stream.read()
            if hashlib.sha256(payload).hexdigest() != SAMPLES[name]:
                msg = f"The downloaded FLEURS sample does not match its pinned SHA-256: {name}"
                raise ValueError(msg)
            payloads[name] = payload
            if len(payloads) == len(SAMPLES):
                break
    if payloads.keys() != SAMPLES.keys():
        msg = "The pinned FLEURS archive prefix did not contain both expected samples"
        raise ValueError(msg)
    return payloads


def download_samples(output_dir: Path) -> Path:
    """Save the pinned originals and manifest, refusing conflicting local files."""
    metadata = _download_prefix(f"{BASE_URL}/dev.tsv", 1024 * 1024)
    if hashlib.sha256(metadata).hexdigest() != METADATA_SHA256:
        msg = "The downloaded FLEURS transcript metadata does not match the pinned revision"
        raise ValueError(msg)
    transcripts = {row[1]: row for row in csv.reader(io.StringIO(metadata.decode("utf-8")), delimiter="\t")}
    payloads = _download_audio()

    rows = []
    for name in SAMPLES:
        info = sf.info(io.BytesIO(payloads[name]))
        transcript = transcripts[name]
        if info.samplerate != SAMPLE_RATE or info.channels != 1 or info.frames != int(transcript[5]):
            msg = f"Expected mono 16 kHz FLEURS audio matching its reference sample count: {name}"
            raise ValueError(msg)
        if not transcript[2].strip():
            msg = f"Missing FLEURS reference transcript: {name}"
            raise ValueError(msg)
        rows.append(
            {
                "audio_filepath": f"audio/{name}",
                "audio_item_id": f"fleurs_hi_in_validation_{transcript[0]}_{Path(name).stem}",
                "source_lang": "hi",
                "duration": info.frames / SAMPLE_RATE,
                "sampling_rate": SAMPLE_RATE,
                "text": transcript[2],
                "dataset": "google/fleurs",
                "dataset_revision": REVISION,
                "license": "CC-BY-4.0",
            }
        )

    output_dir = output_dir.expanduser().resolve()
    files = {output_dir / "audio" / name: payloads[name] for name in SAMPLES}
    manifest = output_dir / "manifest.jsonl"
    files[manifest] = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    for path, payload in files.items():
        if path.is_symlink() or (
            path.exists()
            and (not path.is_file() or path.stat().st_size != len(payload) or path.read_bytes() != payload)
        ):
            msg = f"Refusing to overwrite a different existing file: {path}"
            raise FileExistsError(msg)
    for path, payload in files.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(payload)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/indic_asr"))
    args = parser.parse_args()
    print(download_samples(args.output_dir))
