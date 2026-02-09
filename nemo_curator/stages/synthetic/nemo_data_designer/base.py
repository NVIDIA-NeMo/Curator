# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import time
from dataclasses import dataclass

import data_designer.config as dd
from data_designer.interface import DataDesigner
from loguru import logger

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch


@dataclass
class BaseDataDesignerStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Base class for Nemo Data Designer stage.

    This class provides a base class for Nemo Data Designer stage.
    """
    num_gpus_per_worker: float = 0.0
    config_builder: dd.DataDesignerConfigBuilder | None = None
    data_designer_config_file: str | None = None
    verbose: bool = False

    @property
    def name(self) -> str:
        return "NemoDataDesignerBaseStage"

    @property
    def resources(self) -> Resources:
        return Resources(gpus=self.num_gpus_per_worker)

    def __post_init__(self) -> None:

        # check config_builder and data_designer_config_file
        if self.config_builder is None and self.data_designer_config_file is None:
            msg = "Either 'config_builder' or 'data_designer_config_file' must be set."
            raise ValueError(msg)
        if self.config_builder is not None and self.data_designer_config_file is not None:
            msg = "Only one of 'config_builder' or 'data_designer_config_file' can be set, not both."
            raise ValueError(msg)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Initialize the data designer stage."""

        # read config from file if config_builder is not set
        if self.config_builder is None:
            self.config_builder = dd.DataDesignerConfigBuilder.from_config(self.data_designer_config_file)

        # validate config builder
        self.data_designer = DataDesigner()
        # DEBUGGING
        # not validate here because we haven't load the seed dataset yet
        # self.data_designer.validate(self.config_builder)

        if self.verbose:
            logger.debug("Initialized data designer stage.")

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        num_input_records = len(batch.data)

        # set seed dataframe from batch
        self.config_builder.with_seed_dataset(dd.DataFrameSeedSource(df=batch.data))

        # run preview to get the results
        t1 = time.perf_counter()
        results = self.data_designer.preview(self.config_builder, num_records=num_input_records)
        df = results.dataset
        ndd_running_time = time.perf_counter() - t1

        num_output_records = len(df)
        self._log_metrics(
            {
                "ndd_running_time": ndd_running_time,
                "num_input_records": float(num_input_records),
                "num_output_records": float(num_output_records),
            }
        )

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

# Explicitly export the class
__all__ = ["BaseDataDesignerStage"]
