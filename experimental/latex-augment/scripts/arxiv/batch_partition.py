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
Partition arXiv bulk data tars into indexed tars for latex compilation.

We repartition the data into shards with roughly same size (in bytes) per shard, but keep the documents in the original order.
The output files are meant to be compiled by nvpdftex and the shard size should be small enough to avoid slurm timeouts.
"""

import contextlib
import glob
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections.abc import Iterable

import tqdm

src_path = "/data/arxiv_parsed"
out_basepath = "/data/arxiv_partitioned"
shard_maxsize = 1e9  # ~1000 shards á 1GB


def create_tar_shard(shard_idx_and_paths: tuple[int, list[str]]) -> int:
    """Create a tar file containing all files from arxiv src tars"""
    shard_idx, paths = shard_idx_and_paths
    out_path = os.path.join(out_basepath, f"shard_{shard_idx:06d}.tar")

    num_docs = 0
    with (
        open_tmp(str(out_path) + ".idx", "wb") as index_fd,
        open_tmp(out_path, "wb") as tar_fd,
        tarfile.open(fileobj=tar_fd, mode="w") as tar,
    ):
        for path in paths:
            src_tar = os.path.basename(path)
            for source_dir, arcname in extract_arxiv_tar_dirs(path):
                try:
                    tex_path, docclass, doctype = find_main_tex(source_dir)
                    if not tex_path:
                        continue
                    arxiv_id = extract_arxiv_id(arcname)
                    tex_abs = os.path.join(str(source_dir), tex_path)
                    docmeta = {
                        "tex_path": tex_path,
                        "tex_size": os.path.getsize(tex_abs),
                        "arxiv_src_tar": src_tar,
                        "arxiv_src_file": arcname,
                        "docclass": docclass,
                        "doctype": doctype,
                    }
                    offset = tar_fd.tell()
                    index_fd.write(offset.to_bytes(8, byteorder="little"))
                    add_tar_json(tar, f"{arxiv_id}/__docmeta__.json", docmeta)
                    tar.add(source_dir, arcname=arxiv_id)
                    num_docs += 1
                except (OSError, tarfile.TarError, ValueError) as e:
                    print(f"Error processing directory {source_dir}: {e}")
        # end marker
        offset = tar_fd.tell()
        index_fd.write(offset.to_bytes(8, byteorder="little"))
    return num_docs


def extract_arxiv_tar_dirs(
    tarpath: Path, *, names: set[str] | None = None, delete: bool = True
) -> Iterable[tuple[Path, str]]:
    """Extract document files from arXiv source tars."""
    tempdir = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
    with tempfile.TemporaryDirectory(dir=tempdir, delete=delete) as workdir, tarfile.open(tarpath, "r") as shard_tar:
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


def find_main_tex(source_dir: Path) -> tuple[str | None, str | None, str | None]:
    """Find main .tex path and its docclass and doctype."""
    preferred = [
        "main.tex",
        "ms.tex",
        "paper.tex",
        "manuscript.tex",
        "root.tex",
        "article.tex",
        "arxiv.tex",
        "submission.tex",
    ]

    def score(p: Path) -> tuple[int, int, str]:
        name = p.name
        try:
            pref_idx = preferred.index(name)
        except ValueError:
            pref_idx = len(preferred)
        depth = len(p.relative_to(source_dir).parts)
        return pref_idx, depth, str(p)

    candidates = sorted(source_dir.rglob("*.tex"), key=score)
    for p in candidates:
        try:
            with open(p, "rb") as f:
                data = f.read()
        except OSError:
            continue
        docclass, doctype = parse_docclass_and_doctype(data)
        if docclass and doctype:
            rel = os.path.relpath(str(p), str(source_dir))
            return rel, docclass, doctype
    return None, None, None


def parse_docclass_and_doctype(text: bytes) -> tuple[str | None, str | None]:
    """Parse LaTeX docclass macro and class name."""
    clean = strip_comments(text)
    m = re.search(
        rb"(\\documentclass|\\documentstyle)(?:\s*\[[^\]]*\])?\s*\{\s*([^}\s]+)\s*\}",
        clean,
        re.DOTALL,
    )
    if m:
        macro = m.group(1).decode("ascii", "ignore")
        doctype = m.group(2).decode("ascii", "ignore")
        return macro, doctype
    else:
        return None, None


def strip_comments(text: bytes) -> bytes:
    """Remove LaTeX comments."""
    out: list[bytes] = []
    for line in text.splitlines():
        i = 0
        cut = len(line)
        while i < len(line):
            j = line.find(b"%", i)
            if j == -1:
                break
            if j > 0 and line[j - 1] == 92:
                i = j + 1
                continue
            cut = j
            break
        out.append(line[:cut])
    return b"\n".join(out)


def extract_arxiv_id(arcname: str) -> str:
    """Extract arXiv id from member name like 2306/2306.04957.gz."""
    name = arcname.rsplit("/", 1)[-1]
    if name.endswith(".gz"):
        return name[:-3]
    return os.path.splitext(name)[0]


def add_tar_json(tar: tarfile.TarFile, arcpath: str, data: dict) -> None:
    """Add a JSON file to the tar archive from memory.

    Args:
        tar: TarFile object to add to
        arcpath: Full path within the tar archive (e.g. "paper123/__docmeta__.json")
        data: Python object to serialize as JSON
    """
    json_content = json.dumps(data).encode("utf-8")

    # Create TarInfo for the JSON file
    json_info = tarfile.TarInfo(name=arcpath)
    json_info.size = len(json_content)

    # Add JSON file to tar from memory
    tar.addfile(json_info, fileobj=io.BytesIO(json_content))


@contextlib.contextmanager
def open_tmp(path: str, mode: str) -> Iterable:
    """Write to a temporary file and atomically rename on success."""
    if mode not in ("w", "wb"):
        msg = f"Invalid mode: {mode}"
        raise ValueError(msg)
    tmp_path = str(path) + ".tmp"
    try:
        if os.path.exists(path):
            os.unlink(path)
        with open(tmp_path, mode) as f:
            yield f
        os.rename(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def main() -> None:
    os.makedirs(out_basepath, exist_ok=True)

    paths = sorted(glob.glob(f"{src_path}/*.tar"), key=lambda x: hash(x))
    sizes = [os.path.getsize(x) for x in paths]
    shard_size = 0
    shards = [[]]
    for path, size in zip(paths, sizes, strict=False):
        if shard_size + size > shard_maxsize:
            shards.append([])
            shard_size = 0
        shards[-1].append(path)
        shard_size += size

    num_docs = 0
    with ProcessPoolExecutor() as executor:
        res = executor.map(create_tar_shard, enumerate(shards))
        for nd in tqdm.tqdm(res, total=len(shards), desc="Writing shards"):
            num_docs += nd
    print(f"Processed {num_docs} documents in {len(shards)} shards")


if __name__ == "__main__":
    main()
