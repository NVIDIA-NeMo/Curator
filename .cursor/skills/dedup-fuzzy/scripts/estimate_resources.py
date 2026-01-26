#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Estimate resource requirements for fuzzy deduplication.

This script analyzes an input dataset and estimates the memory, GPU,
and cluster requirements for running FuzzyDeduplicationWorkflow.

Examples:
    # Estimate from Parquet directory
    python estimate_resources.py --input-path /data/text

    # Estimate from JSONL with custom parameters
    python estimate_resources.py \\
        --input-path /data/text \\
        --input-format jsonl \\
        --num-bands 20 \\
        --minhashes-per-band 13
"""
import argparse
import json
import os
import sys
from pathlib import Path


def get_dataset_size(path: str, file_format: str = "parquet") -> dict:
    """Get dataset size information.

    Args:
        path: Path to dataset directory or file
        file_format: File format (parquet or jsonl)

    Returns:
        Dictionary with size information
    """
    p = Path(path)

    if not p.exists():
        return {"error": f"Path does not exist: {path}"}

    extensions = {
        "parquet": {".parquet", ".pq"},
        "jsonl": {".jsonl", ".json", ".ndjson"},
    }

    target_exts = extensions.get(file_format, {f".{file_format}"})

    total_size = 0
    file_count = 0

    if p.is_file():
        total_size = p.stat().st_size
        file_count = 1
    else:
        for f in p.rglob("*"):
            if f.is_file() and f.suffix.lower() in target_exts:
                total_size += f.stat().st_size
                file_count += 1

    return {
        "path": str(path),
        "format": file_format,
        "file_count": file_count,
        "total_bytes": total_size,
        "total_gb": total_size / (1024**3),
    }


def estimate_resources(
    dataset_size_gb: float,
    num_bands: int = 20,
    minhashes_per_band: int = 13,
    avg_doc_size_bytes: int = 2000,
) -> dict:
    """Estimate resource requirements for fuzzy deduplication.

    Args:
        dataset_size_gb: Dataset size in GB
        num_bands: Number of LSH bands
        minhashes_per_band: Hashes per band
        avg_doc_size_bytes: Average document size in bytes

    Returns:
        Dictionary with resource recommendations
    """
    # Estimate document count
    est_doc_count = int((dataset_size_gb * 1024**3) / avg_doc_size_bytes)

    # Memory per document (MinHash signatures + overhead)
    num_hashes = num_bands * minhashes_per_band
    bytes_per_hash = 4  # 32-bit by default
    minhash_memory_per_doc = num_hashes * bytes_per_hash
    overhead_factor = 2.5  # For intermediate data structures

    # Total memory estimate
    base_memory_gb = (est_doc_count * minhash_memory_per_doc * overhead_factor) / (1024**3)

    # LSH shuffle memory (depends on bands_per_iteration)
    # With bands_per_iteration=1, shuffle memory is ~num_bands times lower
    shuffle_memory_multiplier = 1.5

    # Recommendations based on size
    if dataset_size_gb < 10:
        recommendation = {
            "cluster_size": "single_node",
            "nodes": 1,
            "gpus_per_node": 1,
            "memory_per_node_gb": 32,
            "bands_per_iteration": num_bands,  # Can do all at once
            "estimated_time": "< 1 hour",
        }
    elif dataset_size_gb < 100:
        recommendation = {
            "cluster_size": "single_node_multi_gpu",
            "nodes": 1,
            "gpus_per_node": 4,
            "memory_per_node_gb": 64,
            "bands_per_iteration": 5,
            "estimated_time": "1-4 hours",
        }
    elif dataset_size_gb < 500:
        recommendation = {
            "cluster_size": "small_cluster",
            "nodes": 2,
            "gpus_per_node": 4,
            "memory_per_node_gb": 128,
            "bands_per_iteration": 3,
            "estimated_time": "4-12 hours",
        }
    elif dataset_size_gb < 1000:
        recommendation = {
            "cluster_size": "medium_cluster",
            "nodes": 4,
            "gpus_per_node": 8,
            "memory_per_node_gb": 256,
            "bands_per_iteration": 2,
            "estimated_time": "12-24 hours",
        }
    else:
        recommendation = {
            "cluster_size": "large_cluster",
            "nodes": 8,
            "gpus_per_node": 8,
            "memory_per_node_gb": 512,
            "bands_per_iteration": 1,
            "use_64_bit_hash": dataset_size_gb > 2000,
            "estimated_time": "24-72 hours",
        }

    return {
        "dataset": {
            "size_gb": dataset_size_gb,
            "estimated_documents": est_doc_count,
            "estimated_documents_formatted": f"{est_doc_count:,}",
        },
        "memory_estimate": {
            "minhash_memory_gb": base_memory_gb,
            "total_with_shuffle_gb": base_memory_gb * shuffle_memory_multiplier,
            "per_band_iteration_gb": (base_memory_gb * shuffle_memory_multiplier)
            / num_bands,
        },
        "recommendation": recommendation,
        "parameters": {
            "num_bands": num_bands,
            "minhashes_per_band": minhashes_per_band,
            "total_hashes": num_hashes,
            "similarity_threshold": (1 / num_bands) ** (1 / minhashes_per_band),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to input data directory",
    )
    parser.add_argument(
        "--input-format",
        default="parquet",
        choices=["parquet", "jsonl"],
        help="Input file format (default: parquet)",
    )
    parser.add_argument(
        "--num-bands",
        type=int,
        default=20,
        help="Number of LSH bands (default: 20)",
    )
    parser.add_argument(
        "--minhashes-per-band",
        type=int,
        default=13,
        help="Hashes per band (default: 13)",
    )
    parser.add_argument(
        "--avg-doc-size",
        type=int,
        default=2000,
        help="Average document size in bytes (default: 2000)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON only",
    )

    args = parser.parse_args()

    # Get dataset size
    size_info = get_dataset_size(args.input_path, args.input_format)

    if "error" in size_info:
        print(json.dumps(size_info) if args.json else f"❌ {size_info['error']}")
        sys.exit(1)

    if size_info["file_count"] == 0:
        result = {
            "error": f"No {args.input_format} files found in {args.input_path}"
        }
        print(json.dumps(result) if args.json else f"❌ {result['error']}")
        sys.exit(1)

    # Estimate resources
    result = estimate_resources(
        dataset_size_gb=size_info["total_gb"],
        num_bands=args.num_bands,
        minhashes_per_band=args.minhashes_per_band,
        avg_doc_size_bytes=args.avg_doc_size,
    )
    result["input"] = size_info

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        # Human-readable output
        print(f"📊 Dataset Analysis")
        print(f"   Path: {size_info['path']}")
        print(f"   Files: {size_info['file_count']} {size_info['format']} files")
        print(f"   Size: {size_info['total_gb']:.2f} GB")
        print(f"   Est. Documents: {result['dataset']['estimated_documents_formatted']}")
        print()

        print(f"💾 Memory Estimate")
        print(f"   MinHash signatures: {result['memory_estimate']['minhash_memory_gb']:.1f} GB")
        print(f"   With shuffle overhead: {result['memory_estimate']['total_with_shuffle_gb']:.1f} GB")
        print(f"   Per band iteration: {result['memory_estimate']['per_band_iteration_gb']:.1f} GB")
        print()

        rec = result["recommendation"]
        print(f"🖥️  Recommendation: {rec['cluster_size']}")
        print(f"   Nodes: {rec['nodes']}")
        print(f"   GPUs per node: {rec['gpus_per_node']}")
        print(f"   Memory per node: {rec['memory_per_node_gb']} GB")
        print(f"   bands_per_iteration: {rec['bands_per_iteration']}")
        if rec.get("use_64_bit_hash"):
            print(f"   use_64_bit_hash: true (dataset > 2TB)")
        print(f"   Estimated time: {rec['estimated_time']}")
        print()

        params = result["parameters"]
        print(f"⚙️  Parameters")
        print(f"   Total hashes: {params['total_hashes']}")
        print(f"   Similarity threshold: {params['similarity_threshold']:.2f} ({params['similarity_threshold']*100:.0f}%)")


if __name__ == "__main__":
    main()
