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

from typing import TYPE_CHECKING, Any, Literal

from nemo_curator.stages.deduplication.fuzzy.utils import CURATOR_FUZZY_DUPLICATE_GROUP_FIELD
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.stages.deduplication.shuffle_utils.rapidsmpf_shuffler import pylibcudf_to_cudf_dataframe
from nemo_curator.stages.deduplication.shuffle_utils.stage import ShuffleStage
from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.file_utils import get_fs

if TYPE_CHECKING:
    import cudf

DUPLICATE_IDS_SUBDIR = "FuzzyDuplicateIds"


class IdentifyDuplicatesStage(ShuffleStage):
    """
    Stage that generates removal IDs for fuzzy deduplication.
    The approach involves shuffling the data based on the duplicate group field similar to grouping by the group field.
    followed by selecting one document per group.
    Currently the removal strategy is to randomly keep one document per group.

    Parameters
    ----------
    duplicate_group_field
        Column name representing the group id for a document.
    total_nparts
        Total number of output partitions. If None, will be set automatically by the executor.
    output_path
        Path to write output files.
    read_kwargs
        Keyword arguments for cudf.read_parquet method.
    write_kwargs
        Keyword arguments for cudf.to_parquet method.
    rmm_pool_size
        Size of the RMM GPU memory pool in bytes.
        If "auto", the memory pool is set to 90% of the free GPU memory.
        If None, the memory pool is set to 50% of the free GPU memory that can expand if needed.
    spill_memory_limit
        Device memory limit in bytes for spilling to host.
        If "auto", the limit is set to 80% of the RMM pool size.
        If None spilling is disabled.
    enable_statistics
        Whether the underlying rapidsmpf shuffler should collect shuffle statistics.
    """

    _name = "IdentifyDuplicates"

    def __init__(  # noqa: PLR0913
        self,
        duplicate_group_field: str = CURATOR_FUZZY_DUPLICATE_GROUP_FIELD,
        document_id_field: str = CURATOR_DEDUP_ID_STR,
        total_nparts: int | None = None,
        output_path: str = "./",
        read_kwargs: dict[str, Any] | None = None,
        write_kwargs: dict[str, Any] | None = None,
        rmm_pool_size: int | Literal["auto"] | None = "auto",
        spill_memory_limit: int | Literal["auto"] | None = "auto",
        enable_statistics: bool = False,
    ):
        self.duplicate_group_field = duplicate_group_field
        self.document_id_field = document_id_field
        self.output_fs = get_fs(
            output_path, storage_options=read_kwargs.get("storage_options") if read_kwargs is not None else None
        )
        self.output_path = self.output_fs.sep.join([output_path, DUPLICATE_IDS_SUBDIR])
        self.write_kwargs = write_kwargs

        super().__init__(
            shuffle_on=[duplicate_group_field],
            total_nparts=total_nparts,
            output_path=self.output_path,
            read_kwargs=read_kwargs,
            write_kwargs=write_kwargs,
            rmm_pool_size=rmm_pool_size,
            spill_memory_limit=spill_memory_limit,
            enable_statistics=enable_statistics,
        )

    def _get_removal_ids(self, df: "cudf.DataFrame") -> "cudf.DataFrame":
        """
        Get the removal ids for the given dataframe.
        """
        if len(df) == 0:
            return df[[self.document_id_field]]

        removal_ids = df[df[self.duplicate_group_field].duplicated(keep="first")][self.document_id_field]
        removal_ids = removal_ids.sort_values(ignore_index=True)
        return removal_ids.to_frame()

    def process(self, task: FileGroupTask) -> FileGroupTask:
        return super().process(task)

    def ray_stage_spec(self) -> dict[str, Any]:
        return super().ray_stage_spec()

    def read_and_insert(self, task: FileGroupTask) -> FileGroupTask:
        super().read_and_insert(task)
        return task

    def insert_finished(self) -> None:
        super().insert_finished()

    def extract_and_write(self) -> list[FileGroupTask]:
        self._check_actor_obj()
        write_kwargs = self.write_kwargs.copy()
        write_kwargs["index"] = write_kwargs.get("index", False)

        result_tasks = []
        for partition_id, partition in self._actor_obj.extract():
            shuffled_partition_df = pylibcudf_to_cudf_dataframe(partition, column_names=self.output_columns)
            num_groups = shuffled_partition_df[self.duplicate_group_field].nunique()
            removal_ids = self._get_removal_ids(shuffled_partition_df)

            output_file = self.output_fs.sep.join([self.output_path, f"part.{partition_id}.parquet"])
            # If user has not specified row_group_size_rows, set it to the lower of 10% of the number of removal ids or 1M (default) or a minimum of 1k (for small datasets)
            write_kwargs["row_group_size_rows"] = write_kwargs.get(
                "row_group_size_rows", max(1000, min(len(removal_ids) // 10, 1000 * 1000))
            )
            removal_ids.to_parquet(output_file, **write_kwargs)
            result_tasks.append(
                FileGroupTask(
                    task_id=partition_id,
                    dataset_name=self.dataset_name + f"{self.name}",
                    data=[output_file],
                    _metadata={
                        "partition_index": partition_id,
                        "num_groups": num_groups,
                        "num_removal_ids": len(removal_ids),
                    },
                )
            )
        return result_tasks
