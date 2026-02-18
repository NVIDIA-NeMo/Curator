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

from dataclasses import dataclass, field
from typing import Any

from nemo_curator.stages.base import CompositeStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.multimodal.io.readers.webdataset import WebdatasetReaderStage
from nemo_curator.stages.multimodal.utils import (
    DEFAULT_IMAGE_EXTENSIONS,
    DEFAULT_JSON_EXTENSIONS,
    DEFAULT_WEBDATASET_EXTENSIONS,
    require_source_id_field,
    resolve_storage_options,
)
from nemo_curator.tasks import MultiBatchTask, _EmptyTask


@dataclass
class WebdatasetReader(CompositeStage[_EmptyTask, MultiBatchTask]):
    """Composite stage for reading WebDataset shards."""

    file_paths: str | list[str]
    files_per_partition: int | None = None
    blocksize: int | str | None = None
    max_batch_bytes: int | None = None
    read_kwargs: dict[str, Any] = field(default_factory=dict)
    materialize_on_read: bool = False
    file_extensions: list[str] = field(default_factory=lambda: list(DEFAULT_WEBDATASET_EXTENSIONS))
    json_extensions: list[str] = field(default_factory=lambda: list(DEFAULT_JSON_EXTENSIONS))
    image_extensions: list[str] = field(default_factory=lambda: list(DEFAULT_IMAGE_EXTENSIONS))
    source_id_field: str = ""
    sample_id_field: str | None = None
    texts_field: str = "texts"
    images_field: str = "images"
    image_member_field: str | None = None
    name: str = "webdataset_reader"

    def __post_init__(self):
        super().__init__()
        self.source_id_field = require_source_id_field(self.source_id_field)
        self.storage_options = resolve_storage_options(io_kwargs=self.read_kwargs)

    def decompose(self) -> list:
        return [
            FilePartitioningStage(
                file_paths=self.file_paths,
                files_per_partition=self.files_per_partition,
                blocksize=self.blocksize,
                file_extensions=self.file_extensions,
                storage_options=self.storage_options,
            ),
            WebdatasetReaderStage(
                read_kwargs=self.read_kwargs,
                materialize_on_read=self.materialize_on_read,
                max_batch_bytes=self.max_batch_bytes,
                json_extensions=tuple(self.json_extensions),
                image_extensions=tuple(self.image_extensions),
                source_id_field=self.source_id_field,
                sample_id_field=self.sample_id_field,
                texts_field=self.texts_field,
                images_field=self.images_field,
                image_member_field=self.image_member_field,
            ),
        ]
