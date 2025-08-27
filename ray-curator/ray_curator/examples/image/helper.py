# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from __future__ import annotations

import asyncio
import io
import json
import math
import os
import tarfile
from functools import partial
from multiprocessing import Pool
from typing import TYPE_CHECKING

import aiofiles
import aiohttp
import pandas as pd
from loguru import logger
from PIL import Image
from tqdm import tqdm

if TYPE_CHECKING:
    from ray_curator.tasks import ImageObject
    from ray_curator.tasks.image import ImageBatch

# HTTP status codes
HTTP_OK = 200


async def download_image(session: aiohttp.ClientSession, url: str, filename: str, retries: int = 3) -> bool:
    for attempt in range(1, retries + 1):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == HTTP_OK:
                    async with aiofiles.open(filename, mode="wb") as f:
                        await f.write(await response.read())
                    return True
                elif attempt > 1:  # only log on retry attempts, not first try
                    logger.debug(f"[Attempt {attempt}] Failed to download {url}: HTTP status {response.status}")
        except (aiohttp.ClientError, asyncio.TimeoutError, Exception) as e:
            if attempt > 1:  # only log on retry attempts, not first try
                logger.debug(f"[Attempt {attempt}] Failed to download {url}: {e}")

        if attempt < retries:
            await asyncio.sleep(1)  # small delay before retry

    # After all retries failed, log once
    logger.debug(f"All {retries} attempts failed for {url}")
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

            tasks.append(download_image(session, url, jpg_filename, retries=3))

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


def _prepare_metadata_record(
    image_obj: ImageObject,
    new_id: str,
    old_id_col: str | None,
) -> dict:
    """Prepare metadata record for an image object."""
    metadata_record = {
        "id": new_id,
        "original_id": image_obj.image_id,
        "original_path": image_obj.image_path,
    }

    # Preserve original ID in specified column if requested
    if old_id_col:
        metadata_record[old_id_col] = image_obj.image_id

    # Add scores and embeddings to metadata
    if image_obj.aesthetic_score is not None:
        metadata_record["aesthetic_score"] = image_obj.aesthetic_score
    if image_obj.nsfw_score is not None:
        metadata_record["nsfw_score"] = image_obj.nsfw_score
    if image_obj.embedding is not None:
        # Convert embedding to list for JSON serialization
        metadata_record["embedding"] = image_obj.embedding.tolist()
        metadata_record["embedding_dim"] = len(image_obj.embedding)

    # Add original metadata
    if image_obj.metadata:
        metadata_record.update(image_obj.metadata)

    return metadata_record


def _add_caption_to_metadata(image_obj: ImageObject, metadata_record: dict) -> None:
    """Add caption/text to metadata record."""
    if "caption" in image_obj.metadata:
        metadata_record["caption"] = str(image_obj.metadata["caption"])
    elif "text" in image_obj.metadata:
        metadata_record["caption"] = str(image_obj.metadata["text"])
    elif "TEXT" in image_obj.metadata:
        metadata_record["caption"] = str(image_obj.metadata["TEXT"])


def _add_image_to_tar(tar: tarfile.TarFile, image_obj: ImageObject, new_id: str) -> None:
    """Add image data to tar file if available."""
    if image_obj.image_data is not None:
        # Convert numpy array to PIL Image and save as bytes
        image_pil = Image.fromarray(image_obj.image_data)
        image_bytes = _image_to_bytes(image_pil)

        # Add image to tar
        image_info = tarfile.TarInfo(name=f"{new_id}.jpg")
        image_info.size = len(image_bytes.getvalue())
        tar.addfile(image_info, fileobj=image_bytes)


def _add_json_to_tar(tar: tarfile.TarFile, metadata_record: dict, new_id: str) -> None:
    """Add JSON metadata to tar file."""
    json_data = json.dumps(metadata_record, indent=2)
    json_bytes = json_data.encode("utf-8")
    json_info = tarfile.TarInfo(name=f"{new_id}.json")
    json_info.size = len(json_bytes)
    tar.addfile(json_info, fileobj=io.BytesIO(json_bytes))


def save_imagebatch_to_webdataset(
    image_batches: list[ImageBatch],
    output_path: str,
    samples_per_shard: int = 10000,
    max_shards: int = 5,
    old_id_col: str | None = None,
) -> None:
    """
    Save ImageBatch objects to WebDataset format with resharding.

    Args:
        image_batches: List of ImageBatch objects from pipeline output
        output_path: Directory path where the WebDataset should be saved
        samples_per_shard: Number of samples to include in each tar file
        max_shards: Order of magnitude of max shards (for zero-padding filenames)
        old_id_col: If specified, will preserve the original image_id in this column
    """
    os.makedirs(output_path, exist_ok=True)

    # Flatten all ImageObjects from all batches
    all_image_objects = []
    for batch in image_batches:
        all_image_objects.extend(batch.data)

    if not all_image_objects:
        print("No images to save")
        return

    print(f"Processing {len(all_image_objects)} images into {samples_per_shard} samples per shard")

    max_samples_per_shard = math.ceil(math.log10(samples_per_shard))

    # Process images in shards
    shard_id = 0
    for i in range(0, len(all_image_objects), samples_per_shard):
        shard_images = all_image_objects[i:i + samples_per_shard]

        # Create output file paths
        parquet_filename = _name_partition(shard_id, max_shards=max_shards)
        tar_filename = _name_partition(shard_id, max_shards=max_shards, ext="tar")
        parquet_path = os.path.join(output_path, parquet_filename)
        tar_path = os.path.join(output_path, tar_filename)

        # Prepare metadata for parquet
        metadata_records = []

        # Create tar file with images and metadata
        with tarfile.open(tar_path, "w") as tar:
            for sample_idx, image_obj in enumerate(shard_images):
                # Generate new ID combining shard and sample indices
                new_id = _combine_id(
                    shard_id,
                    sample_idx,
                    max_shards=max_shards,
                    max_samples_per_shard=max_samples_per_shard
                )

                # Prepare metadata record for parquet
                metadata_record = _prepare_metadata_record(image_obj, new_id, old_id_col)
                metadata_records.append(metadata_record)

                # Save image data if available and requested
                _add_image_to_tar(tar, image_obj, new_id)

                # Store caption/text in metadata (no separate .txt file)
                _add_caption_to_metadata(image_obj, metadata_record)

                # Add JSON metadata to tar
                _add_json_to_tar(tar, metadata_record, new_id)

        # Save metadata to parquet
        metadata_df = pd.DataFrame(metadata_records)
        metadata_df.to_parquet(parquet_path, index=False)

        print(f"✓ Saved shard {shard_id:0{max_shards}d} with {len(shard_images)} samples")
        print(f"  - Tar file: {tar_filename}")
        print(f"  - Parquet file: {parquet_filename}")

        shard_id += 1

    print(f"\nSuccessfully saved {len(all_image_objects)} images to {shard_id} shards")
    print(f"Output directory: {output_path}")


def _name_partition(
    partition_index: int,
    max_shards: int = 5,
    ext: str = "parquet",
) -> str:
    """Generate partition filename with proper zero-padding."""
    return f"{partition_index:0{max_shards}d}.{ext}"


def _combine_id(shard_id: int, sample_id: int, max_shards: int = 5, max_samples_per_shard: int = 4) -> str:
    """Combine shard and sample IDs into a unique identifier."""
    int_id = sample_id + (10**max_samples_per_shard) * shard_id
    n_digits = max_samples_per_shard + max_shards
    return f"{int_id:0{n_digits}d}"


def _image_to_bytes(image_pil: Image.Image, image_format: str = "JPEG") -> io.BytesIO:
    """Convert PIL Image to BytesIO object for tarfile."""
    buffer = io.BytesIO()
    image_pil.save(buffer, format=image_format)
    buffer.seek(0)
    return buffer
