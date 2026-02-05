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

import logging
import uuid
from dataclasses import dataclass

import torch

import data_designer.config as dd
from data_designer.interface import DataDesigner
from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

logger = logging.getLogger(__name__)

@dataclass
class BaseDataDesignerStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Base class for image filtering stages.

    This class provides a base class for image filtering stages.
    """
    num_gpus_per_worker: float = 0.0
    config_builder: dd.DataDesignerConfigBuilder | None = None
    data_designer_config_file: str | None = None
    verbose: bool = False
    name: str = "data_designer_base"

    def __post_init__(self) -> None:
        if torch.cuda.is_available():
            self.resources = Resources(gpus=self.num_gpus_per_worker)
        else:
            self.resources = Resources()

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Initialize the data designer stage."""

        # check config_builder and data_designer_config_file
        if self.config_builder is None and self.data_designer_config_file is None:
            msg = "Either 'config_builder' or 'data_designer_config_file' must be set."
            raise ValueError(msg)
        if self.config_builder is not None and self.data_designer_config_file is not None:
            msg = "Only one of 'config_builder' or 'data_designer_config_file' can be set, not both."
            raise ValueError(msg)

        # read config from file if config_builder is not set
        if self.config_builder is None:
            self.config_builder = dd.DataDesignerConfigBuilder.from_config(self.data_designer_config_file)

        # validate config builder
        self.data_designer = DataDesigner()
        # DEBUGGING
        # not validate here because we haven't load the seed dataset yet
        # self.data_designer.validate(self.config_builder)

        if self.verbose:
            logger.info("Initialized data designer stage.")

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, batch: DocumentBatch) -> DocumentBatch:

        # DEBUGGING
        print(f"Batch data: {batch.data}")

        # set seed dataframe from batch
        self.config_builder.with_seed_dataset(dd.DataFrameSeedSource(df=batch.data))

        # Generate a unique dataset_name for this batch to avoid race conditions
        # when multiple workers run in parallel (each writes to its own artifact directory)
        unique_dataset_name = f"dataset_{batch.task_id}_{uuid.uuid4().hex[:8]}"
        
        # DEBUGGING
        # think about parallelly write to disk when running self.data_designer.create()
        # results = self.data_designer.preview(self.config_builder, num_records=2)
        # df = results.dataset
        results = self.data_designer.create(self.config_builder, dataset_name=unique_dataset_name)
        df = results.load_dataset()

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

# Explicitly export the class
__all__ = ["BaseDataDesignerStage"]
