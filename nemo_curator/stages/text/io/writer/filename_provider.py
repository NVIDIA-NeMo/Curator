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

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from nemo_curator.tasks import DocumentBatch

from . import utils as writer_utils


class FilenameProvider(ABC):
    """Abstract base class for filename generation strategies.
    
    Similar to Ray Data's FilenameProvider, this allows users to customize
    how output filenames are generated for writers.
    """

    @abstractmethod
    def get_filename(self, task: DocumentBatch, file_extension: str) -> str:
        """Generate a filename for the given task.
        
        Args:
            task: DocumentBatch containing data and metadata
            file_extension: File extension to use (without dot)
            
        Returns:
            Filename with extension (e.g., "my_dataset_abc123.jsonl")
        """


@dataclass
class DefaultFilenameProvider(FilenameProvider):
    """Default filename provider that maintains current behavior.
    
    Uses deterministic hash from source files if available, otherwise UUID.
    """

    def get_filename(self, task: DocumentBatch, file_extension: str) -> str:
        """Generate filename using current default logic."""
        # Get source files from metadata for deterministic naming
        if source_files := task._metadata.get("source_files"):
            filename = writer_utils.get_deterministic_hash(source_files, task.task_id)
        else:
            filename = uuid.uuid4().hex
        
        return f"{filename}.{file_extension}"


@dataclass
class DatasetNameFilenameProvider(FilenameProvider):
    """Filename provider that incorporates dataset name into the filename.
    
    Uses format: {dataset_name}_{hash}.{extension}
    Falls back to default behavior if dataset_name is not available.
    """

    def get_filename(self, task: DocumentBatch, file_extension: str) -> str:
        """Generate filename using dataset name and hash."""
        # Get deterministic part
        if source_files := task._metadata.get("source_files"):
            hash_part = writer_utils.get_deterministic_hash(source_files, task.task_id)
        else:
            hash_part = uuid.uuid4().hex
        
        # Use dataset name if available
        if task.dataset_name:
            filename = f"{task.dataset_name}_{hash_part}"
        else:
            filename = hash_part
            
        return f"{filename}.{file_extension}"


@dataclass
class TemplateFilenameProvider(FilenameProvider):
    """Filename provider that uses a template string for custom naming schemes.
    
    Supports template variables:
    - {dataset_name}: Name of the dataset
    - {task_id}: Task identifier
    - {hash}: Deterministic hash from source files or UUID
    - {extension}: File extension
    
    Example:
        template = "{dataset_name}_{task_id}_{hash}"
        # Generates: "my_dataset_task123_abc456.jsonl"
    """
    
    template: str = "{hash}"
    
    def get_filename(self, task: DocumentBatch, file_extension: str) -> str:
        """Generate filename using template string."""
        # Get deterministic part
        if source_files := task._metadata.get("source_files"):
            hash_part = writer_utils.get_deterministic_hash(source_files, task.task_id)
        else:
            hash_part = uuid.uuid4().hex
        
        # Prepare template variables
        variables = {
            "dataset_name": task.dataset_name or "unknown",
            "task_id": task.task_id,
            "hash": hash_part,
            "extension": file_extension,
        }
        
        # Apply template
        try:
            filename = self.template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Invalid template variable: {e}. Available variables: {list(variables.keys())}")
        
        # Add extension if not already included in template
        if not filename.endswith(f".{file_extension}"):
            filename = f"{filename}.{file_extension}"
            
        return filename