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
from tqdm import tqdm


async def download_image(session: aiohttp.ClientSession, url: str, filename: str) -> bool:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:  # noqa: PLR2004
                async with aiofiles.open(filename, mode="wb") as f:
                    await f.write(await response.read())
                return True
    except (aiohttp.ClientError, asyncio.TimeoutError, Exception):
        # Silently handle download failures - this is expected for some URLs
        pass
    return False


async def process_batch(batch: pd.DataFrame, output_dir: str, batch_num: int) -> None:
    tar_filename = os.path.join(output_dir, f"{batch_num:05d}.tar")
    tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    metadatas = []
    # Set timeout and connection limits for the session
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
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

        results = await asyncio.gather(*tasks, return_exceptions=True)

    with tarfile.open(tar_filename, "w") as tar:
        for i, result in enumerate(results):
            # Check if result is a boolean (successful download) rather than an exception
            if isinstance(result, bool) and result:
                key = f"{batch_num:05d}{i:04d}"
                jpg_base = f"{key}.jpg"
                txt_base = f"{key}.txt"
                json_base = f"{key}.json"
                jpg_tmp = os.path.join(tmp_dir, jpg_base)
                txt_tmp = os.path.join(tmp_dir, txt_base)
                json_tmp = os.path.join(tmp_dir, json_base)

                # Only add files that exist (successful downloads)
                if os.path.exists(jpg_tmp):
                    tar.add(jpg_tmp, arcname=jpg_base)
                    tar.add(txt_tmp, arcname=txt_base)
                    tar.add(json_tmp, arcname=json_base)

    # Clean up temporary files
    for i in range(len(batch)):
        key = f"{batch_num:05d}{i:04d}"
        jpg_tmp = os.path.join(tmp_dir, f"{key}.jpg")
        txt_tmp = os.path.join(tmp_dir, f"{key}.txt")
        json_tmp = os.path.join(tmp_dir, f"{key}.json")

        # Only remove files that exist
        for tmp_file in [jpg_tmp, txt_tmp, json_tmp]:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    # Write parquet
    meta_df = pd.DataFrame(metadatas)
    parquet_path = os.path.join(output_dir, f"{batch_num:05d}.parquet")
    meta_df.to_parquet(parquet_path)


def process_parquet_chunk(chunk: tuple[int, pd.DataFrame], output_dir: str) -> None:
    batch_num, batch = chunk

    asyncio.run(process_batch(batch, output_dir, batch_num))


def download_webdataset(
    parquet_path: str,
    output_dir: str,
    entries_per_tar: int = 10000,
    num_processes: int = 2,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Read the parquet file
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} entries from parquet file")

    # Split the dataframe into chunks for multiprocessing
    chunks = [
        (batch_num, df[i : i + entries_per_tar]) for batch_num, i in enumerate(range(0, len(df), entries_per_tar))
    ]
    print(f"Split into {len(chunks)} chunks of {entries_per_tar} entries each")

    # Use multiprocessing to process chunks in parallel with progress tracking
    with Pool(processes=num_processes) as pool:
        func = partial(process_parquet_chunk, output_dir=output_dir)
        
        # Use tqdm to track progress of chunk processing
        list(tqdm(
            pool.imap(func, chunks), 
            total=len(chunks),
            desc="Processing chunks",
            unit="chunk"
        ))

    tmp_dir = os.path.join(output_dir, "tmp")
    if os.path.exists(tmp_dir):
        os.rmdir(tmp_dir)
