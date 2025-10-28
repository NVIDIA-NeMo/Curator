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
Parse LaTeX documents from arXiv source tars.

The input dataset consists of arXiv source tar files, each containing multiple LaTeX
documents. For each document in the index, this script extracts it, parses it with
fontenc support, and prepares it for translation. One input tar shard produces
one output tar shard, and the shards are processed in parallel across multiple CPU cores.

Source for the incoming bulk data: https://info.arxiv.org/help/bulk_data_s3.html
"""

import csv
import glob
import io
import logging
import os
import shutil
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import tqdm

from latex_augment import LatexDocument, LatexError
from latex_augment.latex_translator import prepare_document


src_path = "/data/arxiv_src"
index_path = "/data/index.csv"
out_path = "/data/arxiv_parsed"


texnames = {}
with open(index_path) as f:
    for shardname, docname, texname, arxiv_id, docclass, doctype in csv.reader(f):
        texnames[docname] = texname


def process_tar(paths):
    inpath, outpath = paths
    docnames = set(texnames.keys())
    with tarfile.open(outpath, "w") as out_tar:
        for docdir, arcname in extract_arxiv_tar_dirs(inpath, names=docnames):
            texname = texnames[arcname]
            try:
                doc = LatexDocument.from_file(docdir / texname)
                # T1 is needed to support all Unicode characters
                doc = doc.with_package("fontenc", ["T1"])
                parsed = prepare_document(doc, document=True)
            except LatexError as e:
                logging.error("%s#%s: %s (ignored)", inpath, arcname, e)
                continue
            except Exception as e:
                logging.exception("%s#%s: %s (ignored)", inpath, arcname, e)
                continue
            with open(docdir / texname, "w") as f:
                f.write(parsed)
            doc_tar_fd = io.BytesIO()
            with tarfile.open(fileobj=doc_tar_fd, mode="w:gz") as doc_tar:
                for name in sorted(os.listdir(docdir)):
                    doc_tar.add(docdir / name, arcname=name)
            info = tarfile.TarInfo(name=arcname)
            info.size = doc_tar_fd.getbuffer().nbytes
            doc_tar_fd.seek(0)
            out_tar.addfile(info, fileobj=doc_tar_fd)


def extract_arxiv_tar_dirs(
    tarpath: Path, *, names=None, delete: bool = True
) -> Iterable[tuple[Path, str]]:
    """Extract document files from arXiv source tars."""
    tempdir = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
    with tempfile.TemporaryDirectory(dir=tempdir, delete=delete) as workdir:
        with tarfile.open(tarpath, "r") as shard_tar:
            for member in shard_tar:
                if not member.isfile():
                    continue
                if not member.name.endswith(".gz"):
                    continue
                if names is not None and member.name not in names:
                    continue
                doc_dir = Path(workdir) / member.name.replace("/", "_")
                doc_tar_fd = shard_tar.extractfile(member)
                with tarfile.open(fileobj=doc_tar_fd, mode="r:gz") as doc_tar:
                    doc_tar.extractall(path=doc_dir)
                yield doc_dir, member.name
                if delete:
                    shutil.rmtree(doc_dir)


if __name__ == "__main__":
    paths = sorted(glob.glob(f"{src_path}/*.tar"))
    outpaths = [p.replace(src_path, out_path) for p in paths]
    os.makedirs(out_path, exist_ok=True)
    with ProcessPoolExecutor() as executor:
        res = executor.map(process_tar, zip(paths, outpaths))
        for _ in tqdm.tqdm(res, total=len(paths)):
            pass
