# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import asyncio
import json
import os
import tarfile
from functools import partial
from multiprocessing import Pool

import aiofiles
import aiohttp
import pandas as pd


async def download_image(session, url, filename):
    async with session.get(url) as response:
        if response.status == 200:
            async with aiofiles.open(filename, mode="wb") as f:
                await f.write(await response.read())
            return True
    return False


async def process_batch(batch, output_dir, batch_num):
    tar_filename = os.path.join(output_dir, f"{batch_num:05d}.tar")
    tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    metadatas = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, (_, row) in enumerate(batch.iterrows()):
            caption = row["TEXT"]
            url = row["URL"]

            key = f"{batch_num:05d}{i:04d}"
            jpg_filename = os.path.join(tmp_dir, f"{key}.jpg")
            txt_filename = os.path.join(tmp_dir, f"{key}.txt")
            json_filename = os.path.join(tmp_dir, f"{key}.json")

            meta = {"url": url, "caption": caption, "key": key}
            metadatas.append(meta)

            tasks.append(download_image(session, url, jpg_filename))

            async with aiofiles.open(txt_filename, mode="w") as f:
                await f.write(caption)

            async with aiofiles.open(json_filename, mode="w") as f:
                await f.write(json.dumps(meta))

        results = await asyncio.gather(*tasks)

    with tarfile.open(tar_filename, "w") as tar:
        for i, success in enumerate(results):
            if success:
                key = f"{batch_num:05d}{i:04d}"
                jpg_base = f"{key}.jpg"
                txt_base = f"{key}.txt"
                json_base = f"{key}.json"
                jpg_tmp = os.path.join(tmp_dir, jpg_base)
                txt_tmp = os.path.join(tmp_dir, txt_base)
                json_tmp = os.path.join(tmp_dir, json_base)

                tar.add(jpg_tmp, arcname=jpg_base)
                tar.add(txt_tmp, arcname=txt_base)
                tar.add(json_tmp, arcname=json_base)

    # Clean up temporary files
    for i in range(len(batch)):
        key = f"{batch_num:05d}{i:04d}"
        jpg_tmp = os.path.join(tmp_dir, f"{key}.jpg")
        txt_tmp = os.path.join(tmp_dir, f"{key}.txt")
        json_tmp = os.path.join(tmp_dir, f"{key}.json")

        os.remove(jpg_tmp)
        os.remove(txt_tmp)
        os.remove(json_tmp)

    # Write parquet
    meta_df = pd.DataFrame(metadatas)
    parquet_path = os.path.join(output_dir, f"{batch_num:05d}.parquet")
    meta_df.to_parquet(parquet_path)


def process_parquet_chunk(chunk, output_dir):
    batch_num, batch = chunk

    asyncio.run(process_batch(batch, output_dir, batch_num))


def download_webdataset(
    parquet_path, output_dir, entries_per_tar=10000, num_processes=2
):
    os.makedirs(output_dir, exist_ok=True)

    # Read the parquet file
    df = pd.read_parquet(parquet_path)

    # Split the dataframe into chunks for multiprocessing
    chunks = [
        (batch_num, df[i : i + entries_per_tar])
        for batch_num, i in enumerate(range(0, len(df), entries_per_tar))
    ]

    # Use multiprocessing to process chunks in parallel
    with Pool(processes=num_processes) as pool:
        func = partial(process_parquet_chunk, output_dir=output_dir)
        pool.map(func, chunks)

    tmp_dir = os.path.join(output_dir, "tmp")
    os.rmdir(tmp_dir)
