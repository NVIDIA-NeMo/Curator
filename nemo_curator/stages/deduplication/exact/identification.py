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

from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR, get_id_generator_actor
from nemo_curator.stages.deduplication.io_utils import DeduplicationIO
from nemo_curator.stages.deduplication.shuffle_utils.rapidsmpf_shuffler import pylibcudf_to_cudf_dataframe
from nemo_curator.stages.deduplication.shuffle_utils.stage import ShuffleStage
from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.file_utils import get_fs

if TYPE_CHECKING:
    import cudf

    from nemo_curator.backends.base import WorkerMetadata


EXACT_DUPLICATE_GROUP_FIELD = "_exact_duplicate_group"


class ExactDuplicateIdentification(DeduplicationIO, ShuffleStage):
    """
    Stage that finds exact duplicates in a given column.

    Parameters
    ----------
    text_field
        Field name representing the field to find duplicates in.
    output_path
        Path to write output files.
    input_filetype
        Type of the input files.
        Must be one of "jsonl" or "parquet". Default is "parquet".
    read_kwargs
        Keyword arguments for cudf.read_parquet method.
    write_kwargs
        Keyword arguments for cudf.to_parquet method.
    assign_id
        Whether to assign a unique id to each document.
    id_field
        Existing id field name if not assigning a new id.
    total_nparts
        Total number of output partitions. If None, will be set automatically by the executor.
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

    _name = "ExactDuplicateIds"

    def __init__(  # noqa: PLR0913
        self,
        text_field: str,
        output_path: str,
        input_filetype: Literal["jsonl", "parquet"] = "parquet",
        read_kwargs: dict[str, Any] | None = None,
        write_kwargs: dict[str, Any] | None = None,
        assign_id: bool = True,
        id_field: str | None = None,
        total_nparts: int | None = None,
        rmm_pool_size: int | Literal["auto"] | None = "auto",
        spill_memory_limit: int | Literal["auto"] | None = "auto",
        enable_statistics: bool = False,
    ):
        if not assign_id and id_field is None:
            msg = "id_field must be provided if assign_id is False"
            raise ValueError(msg)
        elif assign_id and id_field is not None:
            msg = "id_field must be None if assign_id is True"
            raise ValueError(msg)
        # Set instance attributes before parent initialization
        self.text_field = text_field
        self.input_filetype = input_filetype
        self.assign_id_field = assign_id
        self.id_field = id_field if id_field is not None else CURATOR_DEDUP_ID_STR
        self.output_fs = get_fs(
            output_path, storage_options=read_kwargs.get("storage_options") if read_kwargs is not None else None
        )
        self.output_path = self.output_fs.sep.join([output_path, self.name])
        # Initialize using cooperative super() with all parameters
        super().__init__(
            id_generator=None,  # DeduplicationIO parameter
            # ShuffleStage parameters
            shuffle_on=[EXACT_DUPLICATE_GROUP_FIELD],
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
            # return empty dataframe with id column
            return df[[self.id_field]]
        removal_ids = df[df[EXACT_DUPLICATE_GROUP_FIELD].duplicated(keep="first")][self.id_field]
        removal_ids = removal_ids.sort_values(ignore_index=True)
        return removal_ids.to_frame()

    def process(self, task: FileGroupTask) -> FileGroupTask:
        msg = "This is a shuffle stage that does not support the process method."
        raise NotImplementedError(msg)

    def ray_stage_spec(self) -> dict[str, Any]:
        return super().ray_stage_spec()

    def setup(self, _worker_metadata: "WorkerMetadata | None" = None) -> None:
        super().setup(_worker_metadata)
        if self.assign_id_field:
            try:
                self.id_generator = get_id_generator_actor()
            except ValueError as e:
                msg = "Did not find a valid ID generator actor. Please ensure that the ID generator actor was started with from nemo_curator.stages.deduplication.id_generator.create_id_generator_actor()"
                raise ValueError(msg) from e
        else:
            self.id_generator = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return (["data"], [])

    def outputs(self) -> tuple[list[str], list[str]]:
        return (["data"], [])

    def read_and_insert(self, task: FileGroupTask) -> FileGroupTask:
        self._check_actor_obj()

        if self.assign_id_field and self.id_generator is None:
            msg = "ID generator not initialized. Call setup() first."
            raise RuntimeError(msg)
        input_columns = [self.text_field] if self.assign_id_field else [self.text_field, self.id_field]
        if self.input_filetype == "jsonl":
            df = self.read_jsonl(
                filepath=task.data, columns=input_columns, assign_id=self.assign_id_field, **self.read_kwargs
            )
        elif self.input_filetype == "parquet":
            df = self.read_parquet(
                filepath=task.data, columns=input_columns, assign_id=self.assign_id_field, **self.read_kwargs
            )
        else:
            msg = f"Unsupported input filetype: {self.input_filetype}"
            raise ValueError(msg)

        hashed_df = df[[self.id_field]]
        hashed_df[EXACT_DUPLICATE_GROUP_FIELD] = df[self.text_field].hash_values(method="md5")
        self.output_columns = list(hashed_df.columns)
        self.dataset_name = task.dataset_name

        self._actor_obj.insert_chunk(hashed_df, self.output_columns)
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
            num_groups = shuffled_partition_df[EXACT_DUPLICATE_GROUP_FIELD].nunique()
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
                    dataset_name=f"{self.dataset_name}_{self.name}",
                    data=[output_file],
                    _metadata={
                        "partition_index": partition_id,
                        "num_groups": num_groups,
                        "num_removal_ids": len(removal_ids),
                    },
                )
            )
        return result_tasks
