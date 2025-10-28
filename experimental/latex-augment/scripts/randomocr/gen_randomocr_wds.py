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


import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import click
import tqdm
import webdataset as wds
from latex_augment.randomocr import generate_sample


def gen_sample_ids(shard_size, *, shard_id, num_shards, last_digit):
    assert 0 <= shard_id < num_shards, "invalid shard_id"
    start = shard_id * shard_size
    end = start + shard_size
    sample_ids = [10 * i + last_digit for i in range(start, end)]
    return sample_ids


def write_shard(config):
    sample_ids = gen_sample_ids(
        config["num_samples_per_shard"],
        shard_id=config["shard_id"],
        num_shards=config["num_shards"],
        last_digit=config["last_digit"],
    )
    desc = f"{config['shard_id']}/{config['num_shards']}"
    # don't escape unicode-in-json to keep it readable
    encoders = {
        **wds.writer.default_handlers,
        "json": lambda x: json.dumps(x, ensure_ascii=False).encode("utf-8"),
    }
    with wds.TarWriter(config["path"], encoder=encoders) as writer:
        with ProcessPoolExecutor() as executor:
            res = executor.map(partial(generate_sample, config["script"]), sample_ids)
            for sample_id, png, label, latex, _, pdf in tqdm.tqdm(
                res, desc=desc, total=len(sample_ids)
            ):
                sample = {
                    "__key__": f"{sample_id:06d}",
                    "png": png,
                    "doclaynet.json": label,
                    # "tex": latex,
                    # "pdf": pdf,
                }
                writer.write(sample)

    print(f"Wrote {len(sample_ids)} samples: {config['path']}")


# PYTHONPATH=../../src python randomocr_gen_wds.py ascii /data/randomocr_ascii


@click.command()
@click.argument("script")
@click.argument("outdir")
def main(script, outdir):
    num_train_shards = 100
    shards = [
        {
            "path": f"{outdir}/train/shard_{shard_id:06d}.tar",
            "script": script,
            "num_samples_per_shard": 5000,
            "shard_id": shard_id,
            "num_shards": num_train_shards,
            "last_digit": 0,
        }
        for shard_id in range(num_train_shards)
    ]
    shards.insert(
        0,
        {
            "path": f"{outdir}/val/shard_{0:06d}.tar",
            "script": script,
            "num_samples_per_shard": 1000,
            "shard_id": 0,
            "num_shards": 1,
            "last_digit": 1,
        }
    )
    os.makedirs(f"{outdir}/train", exist_ok=True)
    os.makedirs(f"{outdir}/val", exist_ok=True)

    for shard in shards:
        write_shard(shard)

    # add energon metadata non-interactively (FIXME hacky)
    shutil.rmtree(os.path.join(outdir, ".nv-meta"), ignore_errors=True)
    retcode = os.system(
        f"printf 'n\\n' | energon prepare --split-parts 'train:train/.*' --split-parts 'val:val/.*' '{outdir}'"
    )
    assert retcode == 0, "energon prepare failed"

    with open(f"{outdir}/.nv-meta/dataset.yaml", "w") as f:
        f.write(
            """
__class__: OCRWebdataset
__module__: megatron.energon
sample_loader: sample_loader.py:sample_loader
part_filter: sample_loader.py:part_filter
"""
        )

    with open(f"{outdir}/.nv-meta/sample_loader.py", "w") as f:
        f.write(
            """
def sample_loader(raw: dict) -> dict:
    return dict(
        __key__=raw["__key__"],
        image=raw["png"],
        text=None,
        block_boxes=[ann["bbox"] for ann in raw["doclaynet.json"]["ann"]],
        block_classes=[ann["category_id"] for ann in raw["doclaynet.json"]["ann"]],
        block_text=[ann["content"] for ann in raw["doclaynet.json"]["ann"]],
    )

def part_filter(part: str) -> bool:
    return part in ("png", "doclaynet.json")
"""
        )


if __name__ == "__main__":
    main()
