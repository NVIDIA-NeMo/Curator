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
Example demonstrating DocumentSplitter and DocumentJoiner usage.

This example shows how to:
1. Split documents into segments using DocumentSplitter
2. Process segments (e.g., filtering, scoring)
3. Join segments back together using DocumentJoiner
"""

import pandas as pd

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.modules import DocumentJoiner, DocumentSplitter
from nemo_curator.tasks import DocumentBatch


def main():
    print("=" * 80)
    print("DocumentSplitter and DocumentJoiner Example")
    print("=" * 80)

    # Create sample documents with paragraph breaks
    documents = pd.DataFrame({
        "doc_id": [1, 2, 3],
        "text": [
            "Introduction to AI.\n\nArtificial Intelligence is transforming the world.\n\nIt has many applications.",
            "Climate change is real.\n\nWe must act now.\n\nThe future depends on our actions today.",
            "Machine learning basics.\n\nDeep learning is a subset of ML.\n\nNeural networks are powerful tools.",
        ],
        "author": ["Alice", "Bob", "Charlie"],
        "category": ["Technology", "Environment", "Technology"],
    })

    print("\n1. Original Documents:")
    print("-" * 80)
    print(documents)
    print(f"\nTotal documents: {len(documents)}")

    # Create a batch
    batch = DocumentBatch(
        task_id="example_batch",
        dataset_name="example_dataset",
        data=documents,
    )

    # Step 1: Split documents into segments
    print("\n2. Splitting documents by paragraph (\\n\\n separator)...")
    print("-" * 80)
    
    splitter = DocumentSplitter(
        separator="\n\n",
        text_field="text",
        segment_id_field="segment_id",
    )
    
    split_batch = splitter.process(batch)
    split_df = split_batch.to_pandas()
    
    print(split_df)
    print(f"\nTotal segments: {len(split_df)}")
    print(f"Documents were split from {len(documents)} to {len(split_df)} segments")

    # Step 2: You could process individual segments here
    # For example, filter out short segments, score segments, etc.
    print("\n3. Processing segments (example: filtering short segments)...")
    print("-" * 80)
    
    # Filter out segments shorter than 20 characters
    filtered_df = split_df[split_df["text"].str.len() >= 20].copy()
    print(f"Filtered from {len(split_df)} to {len(filtered_df)} segments")
    print(filtered_df)

    filtered_batch = DocumentBatch(
        task_id="filtered_batch",
        dataset_name="example_dataset",
        data=filtered_df,
        _metadata=split_batch._metadata,
        _stage_perf=split_batch._stage_perf,
    )

    # Step 3: Join segments back together
    print("\n4. Joining segments back into documents...")
    print("-" * 80)
    
    joiner = DocumentJoiner(
        separator="\n\n",
        text_field="text",
        segment_id_field="segment_id",
        document_id_field="doc_id",
        drop_segment_id_field=True,
    )
    
    joined_batch = joiner.process(filtered_batch)
    joined_df = joined_batch.to_pandas().sort_values("doc_id").reset_index(drop=True)
    
    print(joined_df)
    print(f"\nFinal documents: {len(joined_df)}")

    # Step 4: Compare original and processed
    print("\n5. Comparison:")
    print("-" * 80)
    for idx, row in joined_df.iterrows():
        doc_id = row["doc_id"]
        original_text = documents[documents["doc_id"] == doc_id]["text"].iloc[0]
        processed_text = row["text"]
        
        print(f"\nDocument {doc_id} by {row['author']} ({row['category']}):")
        print(f"  Original paragraphs: {original_text.count(chr(10)) // 2 + 1}")
        print(f"  Processed paragraphs: {processed_text.count(chr(10)) // 2 + 1}")
        if original_text != processed_text:
            print(f"  Status: Modified (short segments removed)")
        else:
            print(f"  Status: Unchanged")

    print("\n" + "=" * 80)
    print("Example with max_length constraint")
    print("=" * 80)

    # Create documents with length information
    documents_with_length = split_df.copy()
    documents_with_length["length"] = documents_with_length["text"].str.len()

    batch_with_length = DocumentBatch(
        task_id="length_batch",
        dataset_name="example_dataset",
        data=documents_with_length,
    )

    # Join with max_length constraint
    joiner_with_limit = DocumentJoiner(
        separator="\n\n",
        text_field="text",
        segment_id_field="segment_id",
        document_id_field="doc_id",
        max_length=100,  # Limit joined documents to 100 characters
        length_field="length",
        drop_segment_id_field=False,  # Keep segment_id to see how documents were re-split
    )

    limited_batch = joiner_with_limit.process(batch_with_length)
    limited_df = limited_batch.to_pandas()

    print("\nJoined with max_length=100:")
    print("-" * 80)
    print(limited_df[["doc_id", "segment_id", "text", "length"]])
    print(f"\nTotal documents after max_length joining: {len(limited_df)}")
    print("Note: Documents may be re-segmented if joining would exceed max_length")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

