#!/usr/bin/env python3

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

"""
Example demonstrating FilenameProvider usage with NeMo Curator writers.

This example shows how to use different filename providers to control
output file naming schemes, similar to Ray Data's FilenameProvider.
"""

import os
import tempfile
import pandas as pd

from nemo_curator.stages.text.io.writer import (
    JsonlWriter,
    ParquetWriter,
    DefaultFilenameProvider,
    DatasetNameFilenameProvider,
    TemplateFilenameProvider,
)
from nemo_curator.tasks import DocumentBatch


def create_sample_data():
    """Create sample DocumentBatch for demonstration."""
    # Create sample data
    data = [
        {"text": "Hello world", "score": 0.9, "language": "en"},
        {"text": "Goodbye world", "score": 0.8, "language": "en"},
        {"text": "Bonjour monde", "score": 0.7, "language": "fr"},
    ]
    
    df = pd.DataFrame(data)
    
    # Create DocumentBatch
    batch = DocumentBatch(
        task_id="task_001",
        dataset_name="example_dataset",
        data=df,
        _metadata={
            "source_files": ["input1.jsonl", "input2.jsonl"],
            "dummy_key": "dummy_value"
        }
    )
    
    return batch


def demonstrate_filename_providers():
    """Demonstrate different filename provider options."""
    
    print("=== NeMo Curator FilenameProvider Demo ===\n")
    
    # Create sample data
    batch = create_sample_data()
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Output directory: {temp_dir}\n")
        
        # 1. Default behavior (backward compatible)
        print("1. Default FilenameProvider (backward compatible):")
        default_dir = os.path.join(temp_dir, "default")
        os.makedirs(default_dir)
        
        writer = JsonlWriter(path=default_dir)
        writer.setup()
        result = writer.process(batch)
        
        filename = os.path.basename(result.data[0])
        print(f"   Generated filename: {filename}")
        print(f"   Uses deterministic hash from source files\n")
        
        # 2. Dataset name in filename
        print("2. DatasetNameFilenameProvider:")
        dataset_dir = os.path.join(temp_dir, "dataset")
        os.makedirs(dataset_dir)
        
        provider = DatasetNameFilenameProvider()
        writer = JsonlWriter(path=dataset_dir, filename_provider=provider)
        writer.setup()
        result = writer.process(batch)
        
        filename = os.path.basename(result.data[0])
        print(f"   Generated filename: {filename}")
        print(f"   Includes dataset name '{batch.dataset_name}' in filename\n")
        
        # 3. Custom template
        print("3. TemplateFilenameProvider with custom pattern:")
        template_dir = os.path.join(temp_dir, "template")
        os.makedirs(template_dir)
        
        provider = TemplateFilenameProvider(template="data_{dataset_name}_{task_id}")
        writer = ParquetWriter(path=template_dir, filename_provider=provider)
        writer.setup()
        result = writer.process(batch)
        
        filename = os.path.basename(result.data[0])
        print(f"   Template: 'data_{{dataset_name}}_{{task_id}}'")
        print(f"   Generated filename: {filename}")
        print(f"   Full control over naming scheme\n")
        
        # 4. Advanced template with all variables
        print("4. Advanced TemplateFilenameProvider:")
        advanced_dir = os.path.join(temp_dir, "advanced")
        os.makedirs(advanced_dir)
        
        provider = TemplateFilenameProvider(
            template="{dataset_name}_batch_{task_id}_{hash}.{extension}"
        )
        writer = JsonlWriter(path=advanced_dir, filename_provider=provider)
        writer.setup()
        result = writer.process(batch)
        
        filename = os.path.basename(result.data[0])
        print(f"   Template: '{{dataset_name}}_batch_{{task_id}}_{{hash}}.{{extension}}'")
        print(f"   Generated filename: {filename}")
        print(f"   Uses all available template variables\n")
        
        print("=== Available Template Variables ===")
        print("- {dataset_name}: Name of the dataset")
        print("- {task_id}: Unique task identifier") 
        print("- {hash}: Deterministic hash from source files or UUID")
        print("- {extension}: File extension (e.g., 'jsonl', 'parquet')")
        print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demonstrate_filename_providers()