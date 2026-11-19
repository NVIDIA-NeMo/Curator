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

"""Stage the per-language AI4Bharat IndicConformer checkpoints in a Hugging Face cache.

The ``ai4bharat/indicconformer_stt_<lang>_hybrid_ctc_rnnt_large`` repositories need a Hugging Face login that accepted their
terms. Without one, this script downloads each ``.nemo`` file from AI4Bharat's public mirror and accepts it only when its size
and SHA-256 equal the values Hugging Face records for the repository's current revision. The file is laid out as a cache
snapshot, so ``HF_HOME=<hf-home> HF_HUB_OFFLINE=1`` resolves ``ai4bharat/indicconformer_stt_<lang>_hybrid_ctc_rnnt_large``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from loguru import logger

REPO_TEMPLATE = "ai4bharat/indicconformer_stt_{lang}_hybrid_ctc_rnnt_large"
FILE_TEMPLATE = "indicconformer_stt_{lang}_hybrid_rnnt_large.nemo"
MIRROR_TEMPLATE = "https://objectstore.e2enetworks.net/indicconformer/models/" + FILE_TEMPLATE
LANGUAGES = "as,bn,brx,doi,gu,hi,kn,kok,ks,mai,ml,mni,mr,ne,or,pa,sa,sat,sd,ta,te,ur"
CHUNK_BYTES = 8 * 1024 * 1024


def _recorded(repo: str, file_name: str) -> tuple[str, int, str]:
    """(revision, size, sha256) Hugging Face records for the repository's ``.nemo`` file."""
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=60) as response:
        info = json.load(response)
    sibling = next(s for s in info["siblings"] if s["rfilename"] == file_name)
    return info["sha"], int(sibling["size"]), sibling["lfs"]["sha256"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def stage_language(hf_home: Path, lang: str) -> None:
    repo, file_name = REPO_TEMPLATE.format(lang=lang), FILE_TEMPLATE.format(lang=lang)
    revision, size, sha256 = _recorded(repo, file_name)
    root = hf_home / "hub" / f"models--{repo.replace('/', '--')}"
    blob, snapshot = root / "blobs" / sha256, root / "snapshots" / revision / file_name
    if blob.is_file() and blob.stat().st_size == size and snapshot.exists():
        logger.info(f"{repo}: already staged")
        return
    for directory in (blob.parent, snapshot.parent, root / "refs"):
        directory.mkdir(parents=True, exist_ok=True)
    partial = blob.with_suffix(".partial")
    urllib.request.urlretrieve(MIRROR_TEMPLATE.format(lang=lang), partial)  # noqa: S310
    if partial.stat().st_size != size or _sha256(partial) != sha256:
        partial.unlink()
        msg = f"{repo}: the mirror file differs from the size or SHA-256 Hugging Face records at revision {revision}"
        raise RuntimeError(msg)
    partial.replace(blob)
    snapshot.unlink(missing_ok=True)
    snapshot.symlink_to(Path("..") / ".." / "blobs" / sha256)
    (root / "refs" / "main").write_text(revision)
    logger.success(f"{repo}: staged {size / 1e6:.0f} MB, sha256 {sha256[:12]} at revision {revision[:8]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-home", type=Path, required=True, help="Hugging Face home; checkpoints land in <hf-home>/hub"
    )
    parser.add_argument("--languages", default=LANGUAGES, help="comma-separated language codes")
    args = parser.parse_args()
    for lang in args.languages.split(","):
        stage_language(args.hf_home, lang)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
