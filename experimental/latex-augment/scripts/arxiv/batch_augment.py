# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Apply visual augmentations to LaTeX documents.

The input dataset is partitioned as tar files, with each shard containing multiple
LaTeX documents. This script applies random visual transformations such as random
fonts, colors, layouts, and spacing to increase training data diversity. One input tar
shard produces one output tar shard, and the shards are processed in parallel across
multiple CPU cores. Augmentations are deterministic and seeded by filename.
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor
import glob
import hashlib
import json
import sys
import random
import shutil
import tarfile
import tempfile
import tqdm
from pathlib import Path
from typing import Iterable

from latex_augment import LatexDocument
from latex_augment import transforms as T

lang = sys.argv[1]

basepath_in = f"/data/arxiv_translated_{lang}"
basepath_out = f"/data/arxiv_translated_{lang}_augmented"

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.WARNING)


augment = T.Compose(
    [
        T.RandomPageSizeAndMargins(p=0.3),
        T.RandomLineSpacing(p=0.3),
        T.RandomWordSpacing(p=0.3),
        T.RandomLetterSpacing(p=0.3),
        T.RandomFontSize(p=0.3),
        T.RandomTextAlignment(p=0.3),
        T.RandomColumnLayout(p=0.3),
        T.RandomSubsectionColumnLayout(p=0.3),
        T.RandomFloatRotation(p=0.1),
        T.RandomTableColumnSeparators(p=0.3),
        T.RemoveBibliography(p=0.3),
        T.RemoveCaptions(p=0.3),
        T.RemoveHeadings(p=0.3),
        T.RemovePageNumbers(p=0.3),
        T.RandomFont(p=1.0),
        T.RandomSubsectionTextColor(p=0.3),
        T.RandomSepiaPageColor(p=0.3),
        T.RandomPageColor(p=0.3),
        # don't do inverted colors because texts end up black-on-black
        # T.RandomInvertedColors(p=0.3),
        T.RandomPageBackground(p=0.3),
        T.RandomTextColor(p=0.3),
    ],
    # scale probs such that there's a 10% chance of no augmentation
    p_any=0.9,
)


def process_shard(shard_path):
    """Process one input shard file."""
    shard_name = os.path.basename(shard_path)
    out_shard = f"{basepath_out}/{shard_name}"
    failed = total = 0

    with tarfile.open(out_shard, "w") as out_tar:
        for dirname, arcname in extract_tar_dirs(shard_path):
            with open(dirname / "__docmeta__.json", "rb") as f:
                docmeta = json.load(f)
            tex_path = docmeta["tex_path"]

            total += 1
            try:
                # deterministic augmentations seeded with filename
                seed = md5_int32(f"{dirname}/{tex_path}:augment")
                rng = random.Random(seed)
                doc = LatexDocument.from_file(dirname / tex_path)
                # T1 is needed to support all Unicode characters
                doc = doc.with_package("fontenc", ["T1"])
                doc = augment(doc, rng=rng)
                with open(dirname / tex_path, "wb") as f:
                    f.write(doc.source)
            except Exception as e:
                logging.exception(
                    "%s %s/%s seed %d: %s (ignored)",
                    shard_name,
                    dirname,
                    tex_path,
                    seed,
                    e,
                )
                failed += 1
                print(f"fail rate: {failed} / {total}", file=sys.stderr)
                continue
            out_tar.add(dirname, arcname=arcname)

    print(f"{shard_name}: {total - failed} augmented, {failed} failed", file=sys.stderr)


def extract_tar_dirs(
    tarpath: Path, *, delete: bool = True,
) -> Iterable[tuple[Path, str]]:
    """Extract each tar top level directory."""
    tempdir = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
    with tempfile.TemporaryDirectory(dir=tempdir, delete=delete) as workdir:
        current_dir = None
        with tarfile.open(tarpath, "r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                if "/" not in member.name:
                    continue
                top_dir, _ = member.name.split("/", 1)
                if top_dir != current_dir:
                    if current_dir is not None:
                        yield Path(workdir) / current_dir, current_dir
                        if delete:
                            shutil.rmtree(Path(workdir) / current_dir)
                    current_dir = top_dir
                tar.extract(member, path=workdir)
            if current_dir is not None:
                yield Path(workdir) / current_dir, current_dir


def md5_int32(text):
    """Hash text into signed 32-bit integer"""
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def main():
    shard_paths = glob.glob(f"{basepath_in}/shard_*.tar")
    os.makedirs(basepath_out, exist_ok=True)
    # for path in shard_paths:
    #     process_shard(path)
    with ProcessPoolExecutor() as executor:
        res = executor.map(process_shard, shard_paths)
        for _ in tqdm.tqdm(res, total=len(shard_paths), desc="Processing shards"):
            pass


if __name__ == "__main__":
    main()
