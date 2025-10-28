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

r"""
Distributed LaTeX document translation using tritonserver.

The input dataset is partitioned as tar files, with each shard containing multiple English
LaTeX documents. One input tar shard produces one output tar shard, and the shards are
processed in parallel across multiple nodes and GPUs. The shards are assigned to GPU tasks
in round robin order. You should run the script with enough nodes to process all input files.
"""

import atexit
import glob
import hashlib
import json
import logging
import os
import requests
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable

import tqdm
from tritonclient.utils import InferenceServerException

from latex_augment.latex_translator import LatexError, translate_split_document
from latex_augment.triton_translator import TritonTranslator

##### CONFIGURATION #####

lang = sys.argv[1]
node_id = int(sys.argv[2])

basepath_in = "/data/arxiv_partitioned"
basepath_out = f"/data/arxiv_translated_{lang}"
triton_model_repo = "/workspace/tritonserver_data/enc_dec_ifb"
subsample = 1/3

assert (
    "LOCAL_RANK" in os.environ and "LOCAL_WORLD_SIZE" in os.environ
), "unknown GPU configuration"
local_rank = int(os.environ.get("LOCAL_RANK"))
local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE"))
server_port = 24680 + local_rank
metrics_port = 24780 + local_rank
server_address = f"localhost:{server_port}"
num_workers = 4

# parallelized over many independent single-node jobs
task_id = node_id * local_world_size + local_rank

logging.basicConfig(
    format=f"%(asctime)s {task_id}: %(message)s",
    level=logging.WARNING,
)


def update_log_format(tex_path):
    formatter = logging.getLogger().handlers[0].formatter
    formatter._fmt = f"%(asctime)s {task_id}: {tex_path}: %(message)s"


##### START TRITON SERVER #####


def start_triton_server():
    server_cmd = [
        "mpirun",
        "--allow-run-as-root",
        "-n",
        "1",
        "/opt/tritonserver/bin/tritonserver",
        f"--model-repository={triton_model_repo}",
        f"--grpc-port={server_port}",
        "--allow-http=false",
        f"--metrics-port={metrics_port}",
        "--disable-auto-complete-config",
        "--log-warning=false",
        f"--backend-config=python,shm-region-prefix-name=translate{local_rank}",
    ]

    print("running:", " ".join(server_cmd))
    proc = subprocess.Popen(
        server_cmd,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(local_rank)},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(5)
    return proc


# mitigate repeated hallucinations with repetition penalty, e.g.
# 32、16、16、8、8、8、8、8、8、8、8、12、12、12、12、12、12、12、12、12、...
repetition_penalty = 1.5


def init_worker():
    global triton_client

    logging.basicConfig(
        format=f"%(asctime)s {task_id}: %(message)s",
        level=logging.WARNING,
    )

    manager = TritonTranslator(
        lang, repetition_penalty=repetition_penalty, server_address=server_address
    )
    triton_client = manager.__enter__()
    atexit.register(manager.__exit__, None, None, None)


def translate(dirs):
    docdir, arcname = dirs
    # print("worker", os.getpid(), "translating", arcname)
    global triton_client
    try:
        with open(f"{docdir}/__docmeta__.json", "rb") as f:
            docmeta = json.load(f)
        update_log_format(arcname)
        tex_path = f"{docdir}/{docmeta['tex_path']}"
        with open(tex_path, "r", encoding="utf8") as f:
            tagged_latex = f.read()
        translated_tagged_latex = "".join(
            translate_split_document(
                tagged_latex, keep_tags=True, lang=lang, translator=triton_client
            )
        )
        translated_latex = (
            translated_tagged_latex
            .replace("<s>", "")
            .replace("</s>", "")
            .replace("<latex>", "")
            .replace("</latex>", "")
        )
        with open(tex_path, "w", encoding="utf8") as f:
            f.write(translated_latex)
        if tex_path.endswith(".tex"):
            with open(tex_path.replace(".tex", "_sentences.tex"), "w", encoding="utf8") as f:
                f.write(translated_tagged_latex)
        return docdir, arcname
    except InferenceServerException as e:
        logging.exception("%s: %s", arcname, e)
        raise
    except LatexError as e:
        # reduce log spam
        logging.error("%s: %s (ignored)", arcname, e)
        shutil.rmtree(docdir, ignore_errors=True)
        return None, None
    except Exception as e:
        logging.exception("%s: %s (ignored)", arcname, e)
        shutil.rmtree(docdir, ignore_errors=True)
        return None, None


def extract_tar_dirs(
    tarpath: Path, *, delete: bool = True, subsample: float = 1.0
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
                if md5_unif(f"{top_dir}:{lang}:sample") > subsample:
                    continue
                if top_dir != current_dir:
                    if current_dir is not None:
                        yield Path(workdir) / current_dir, current_dir
                        if delete:
                            shutil.rmtree(Path(workdir) / current_dir)
                    current_dir = top_dir
                tar.extract(member, path=workdir)
            if current_dir is not None:
                yield Path(workdir) / current_dir, current_dir


def query_metrics(port: int) -> tuple[int, int]:
    response = requests.get(f"http://localhost:{port}/metrics")
    input_token_count = output_token_count = None
    for line in response.text.split("\n"):
        if line.startswith("#"):
            continue
        value = line.split(" ")[-1]
        if line.startswith('nv_llm_output_token_len_sum{model="tensorrt_llm",response_metric_type="total_output_tokens"'):
            output_token_count = int(value)
        if line.startswith('nv_llm_input_token_len_sum{model="tensorrt_llm",response_metric_type="total_input_tokens"'):
            input_token_count = int(value)
    return input_token_count, output_token_count


def md5_unif(text):
    """Hash text into uniform range [0, 1)"""
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / (1 << 32)


if __name__ == "__main__":
    paths = sorted(glob.glob(f"{basepath_in}/*.tar"))
    server_process = start_triton_server()
    os.makedirs(basepath_out, exist_ok=True)
    inpath = paths[task_id]
    num_docs = (os.path.getsize(inpath + ".idx") // 8) - 1
    num_docs = int(num_docs * subsample)
    outpath = basepath_out.rstrip("/") + "/" + inpath.split("/")[-1]
    success = failure = 0
    start_time = time.time()
    with tarfile.open(outpath + ".tmp", "w") as tar:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker,
            mp_context=get_context("spawn"),
        ) as executor:
            progress = tqdm.tqdm(total=num_docs, desc=f"{task_id}")
            docpaths = extract_tar_dirs(inpath, delete=False, subsample=subsample)
            for docpath, arcname in executor.map(translate, docpaths):
                if docpath is not None:
                    tar.add(docpath, arcname=arcname)
                    shutil.rmtree(docpath)
                    success += 1
                else:
                    failure += 1
                progress.update(1)
                if success % 10 == 0:
                    input_token_count, output_token_count = query_metrics(metrics_port)
                    throughput = output_token_count / (time.time() - start_time)
                    print(f"\n{task_id}: Output token throughput: {throughput:.2f} tokens/s")
    shutil.move(outpath + ".tmp", outpath)
    server_process.terminate()
    print(f"{task_id}: success: {success}, failure: {failure}")
