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

import io
import math
import os
import tarfile
from functools import partial
from typing import List, Optional

import dask.dataframe as dd
import dask_cudf
import fsspec
import numpy as np
import pandas as pd
from fsspec.core import open_files


class ImageTextPairDataset:
    def __init__(
        self, path: str, metadata: dd.DataFrame, tar_files: List[str], id_col: str
    ) -> None:
        self.path = path
        self.metadata = metadata
        self.tar_files = tar_files
        self.id_col = id_col

    @classmethod
    def from_webdataset(cls, path: str, id_col: str):
        metadata = dask_cudf.read_parquet(path)
        metadata = metadata.map_partitions(cls._sort_partition, id_col=id_col)

        tar_files = cls._get_tar_files(path)

        return cls(path, metadata, tar_files, id_col)

    @staticmethod
    def _sort_partition(partition, id_col):
        return partition.sort_values(id_col).reset_index(drop=True)

    @staticmethod
    def _get_tar_files(path: str) -> List[str]:
        glob_str = os.path.join(path, "*.tar")
        # open_files doesn't actually open a file descriptor
        tar_files = [file.path for file in open_files(glob_str)]

        return tar_files

    @staticmethod
    def _name_partition(
        partition_index: int,
        temp: bool = False,
        max_shards: int = 5,
        ext: str = "parquet",
    ) -> str:
        if temp:
            prefix = "temp_"
        else:
            prefix = ""

        return f"{prefix}{partition_index:0{max_shards}d}.{ext}"

    def save_metadata(
        self, path: Optional[str] = None, columns: Optional[List[str]] = None
    ) -> None:
        if path is None:
            path = self.path

        if columns is None:
            metadata = self.metadata
        else:
            metadata = self.metadata[columns]

        metadata.to_parquet(path, name_function=self._name_partition)

    @staticmethod
    def _filter_valid_members(members, valid_ids):
        def filter_members(member):
            full_id = member.name.split(".")[0]
            sample_id = int(full_id)

            return sample_id in valid_ids

        return list(filter(filter_members, members))

    def _get_eligible_samples(self, output_path: str, samples_per_shard: int):
        parquet_glob_str = os.path.join(output_path, "temp_*.parquet")
        tar_glob_str = os.path.join(self.path, "*.tar")
        parquet_files = open_files(parquet_glob_str)
        tar_files = open_files(tar_glob_str)

        curr_df = None
        total_tar_samples = []
        for parquet_file, tar_file in zip(parquet_files, tar_files):
            with parquet_file as f:
                shard_df = pd.read_parquet(f)

            # Get all the samples associated with this dataframe from the tar file
            valid_member_ids = set(map(int, shard_df[self.id_col]))
            with tar_file as f:
                tar = tarfile.open(fileobj=f)
                valid_members = self._filter_valid_members(
                    tar.getmembers(), valid_member_ids
                )
                valid_members.sort(key=lambda x: x.name)
                tar_samples = [
                    (member, tar.extractfile(member).read()) for member in valid_members
                ]

            if len(tar_samples) % len(shard_df) != 0:
                raise RuntimeError(
                    f"Tarfile {tar_file.path} entries {len(tar_samples)} are not a multiple of the number of samples {len(shard_df)}"
                )

            entries_per_sample = int(len(tar_samples) / len(shard_df))

            # Concat the dataframe and tar file samples
            if curr_df is not None:
                curr_df = pd.concat([curr_df, shard_df], ignore_index=True)
            else:
                curr_df = shard_df
            total_tar_samples.extend(tar_samples)

            # Delete the temp shard
            parquet_file.fs.delete(parquet_file.path)

            # While there are enough samples, yield a slice and discard it
            while len(curr_df) >= samples_per_shard:
                yield (
                    curr_df.iloc[:samples_per_shard].copy(),
                    total_tar_samples[: samples_per_shard * entries_per_sample],
                )
                curr_df = curr_df.iloc[samples_per_shard:]
                total_tar_samples = total_tar_samples[
                    samples_per_shard * entries_per_sample :
                ]

        # Return the remaining df and samples
        yield curr_df, total_tar_samples

    @staticmethod
    def combine_id(shard_id, sample_id, max_shards=5, max_samples_per_shard=4) -> str:
        int_id = sample_id + (10**max_samples_per_shard) * shard_id
        n_digits = max_samples_per_shard + max_shards
        combined_id = f"{int_id:0{n_digits}d}"
        return combined_id

    def split_id(combined_id: str, max_shards=5):
        return int(combined_id[:max_shards]), int(combined_id[max_shards:])

    def to_webdataset(
        self,
        path: str,
        filter_column: str,
        samples_per_shard: int = 10000,
        max_shards=5,
        old_id_col=None,
    ) -> None:
        max_samples_per_shard = math.ceil(math.log10(samples_per_shard))
        filtered_metadata = self.metadata[self.metadata[filter_column]]

        temp_name_fn = partial(self._name_partition, temp=True, max_shards=max_shards)
        filtered_metadata.to_parquet(path, name_function=temp_name_fn)

        for shard_id, (shard_df, shard_tar) in enumerate(
            self._get_eligible_samples(path, samples_per_shard)
        ):
            output_parquet_base = self._name_partition(shard_id, max_shards=max_shards)
            output_tar_base = self._name_partition(
                shard_id, max_shards=max_shards, ext="tar"
            )
            output_parquet_path = os.path.join(path, output_parquet_base)
            output_tar_path = os.path.join(path, output_tar_base)
            output_parquet_file = fsspec.open(output_parquet_path, mode="wb")
            output_tar_file = fsspec.open(output_tar_path, mode="wb")

            # Change the id on the parquet files
            if old_id_col:
                shard_df[old_id_col] = shard_df[self.id_col]

            new_ids = np.arange(len(shard_df))
            convert_ids = partial(
                self.combine_id,
                shard_id,
                max_shards=max_shards,
                max_samples_per_shard=max_samples_per_shard,
            )
            shard_df[self.id_col] = list(map(convert_ids, new_ids))
            with output_parquet_file as f:
                shard_df.to_parquet(f, index=False)

            members_per_sample = len(shard_tar) / len(shard_df)
            with output_tar_file as f:
                tar = tarfile.open(fileobj=f, mode="w")
                for i, (member, data) in enumerate(shard_tar):
                    # Rename the each member to match the new id
                    sample_id = int(i // members_per_sample)
                    member_id = self.combine_id(
                        shard_id,
                        sample_id,
                        max_shards=max_shards,
                        max_samples_per_shard=max_samples_per_shard,
                    )
                    extension = member.name.split(".")[-1]
                    member.name = f"{member_id}.{extension}"

                    tar.addfile(member, io.BytesIO(data))
            print(
                f"Finished writing shard {self._name_partition(shard_id, ext='tar')} with "
                f"parquet length {len(shard_df)} and tar length {len(shard_tar)}"
            )
