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

"""Generate sample data for testing NeMo Curator pipelines.

Usage:
    python generate_sample_data.py --output-dir sample_data --format jsonl --num-docs 100
    python generate_sample_data.py --output-dir sample_data --format parquet --num-docs 100
"""

import argparse
import json
import random
from pathlib import Path

# Sample good-quality paragraphs (will pass filters)
GOOD_PARAGRAPHS = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
    "Natural language processing is a field of computer science and artificial intelligence concerned with the interactions between computers and human languages. It involves programming computers to process and analyze large amounts of natural language data.",
    "Data curation is the process of organizing, integrating, and maintaining data collected from various sources. It involves annotation, publication, and presentation of the data such that the value of the data is maintained over time.",
    "The transformer architecture has revolutionized the field of natural language processing since its introduction. Unlike recurrent neural networks that process sequences sequentially, transformers use self-attention mechanisms to process all positions in parallel.",
    "Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. In machine learning, we use gradient descent to update the parameters of our model.",
    "Deep learning is a class of machine learning algorithms that uses multiple layers to progressively extract higher-level features from the raw input. For example, in image processing, lower layers may identify edges, while higher layers may identify concepts relevant to a human.",
    "Convolutional neural networks are a class of deep neural networks most commonly applied to analyzing visual imagery. They use a mathematical operation called convolution in at least one of their layers instead of general matrix multiplication.",
    "Recurrent neural networks are a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence. This allows them to exhibit temporal dynamic behavior and process sequences of inputs.",
    "Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize the notion of cumulative reward. It differs from supervised learning in not needing labeled input data.",
    "Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task. It is a popular approach in deep learning where pre-trained models are used as the starting point.",
]

# Short texts (will be filtered by WordCountFilter)
SHORT_TEXTS = [
    "Too short.",
    "Brief note.",
    "This is short.",
    "Not enough words here.",
    "Minimal content.",
    "Just a few words.",
    "Very brief text.",
    "Short document.",
    "Tiny snippet.",
    "Small text block.",
]

# High symbol ratio texts (will be filtered by NonAlphaNumericFilter)
SYMBOL_TEXTS = [
    "###$$$%%%^^^&&&***!!!@@@" * 20 + " some words " + "###$$$%%%^^^&&&" * 10,
    "!@#$%^&*()_+{}|:<>?" * 30,
    "~~~~~~!!!!!!######$$$$$$%%%%%%" * 15 + " text " + "^^^^^^&&&&&&" * 10,
    "<<<<<>>>>>====++++" * 25 + " words " + "----____" * 20,
    "@@@###$$$%%%^^^&&&***!!!" * 20,
]


def generate_good_document(doc_id: int) -> dict:
    """Generate a good quality document that should pass filters."""
    # Combine 3-5 random paragraphs
    num_paragraphs = random.randint(3, 5)
    paragraphs = random.choices(GOOD_PARAGRAPHS, k=num_paragraphs)
    text = " ".join(paragraphs)
    return {"id": f"doc_{doc_id:04d}", "text": text}


def generate_short_document(doc_id: int) -> dict:
    """Generate a short document that should be filtered."""
    text = random.choice(SHORT_TEXTS)
    return {"id": f"doc_{doc_id:04d}", "text": text}


def generate_symbol_document(doc_id: int) -> dict:
    """Generate a high-symbol document that should be filtered."""
    text = random.choice(SYMBOL_TEXTS)
    return {"id": f"doc_{doc_id:04d}", "text": text}


def generate_dataset(num_docs: int, good_ratio: float = 0.7) -> list[dict]:
    """Generate a mixed dataset.

    Args:
        num_docs: Total number of documents
        good_ratio: Ratio of good documents (rest split between short and symbol)

    Returns:
        List of document dictionaries
    """
    documents = []
    num_good = int(num_docs * good_ratio)
    num_bad = num_docs - num_good
    num_short = num_bad // 2
    num_symbol = num_bad - num_short

    doc_id = 1

    # Generate good documents
    for _ in range(num_good):
        documents.append(generate_good_document(doc_id))
        doc_id += 1

    # Generate short documents
    for _ in range(num_short):
        documents.append(generate_short_document(doc_id))
        doc_id += 1

    # Generate symbol documents
    for _ in range(num_symbol):
        documents.append(generate_symbol_document(doc_id))
        doc_id += 1

    # Shuffle
    random.shuffle(documents)
    return documents


def save_jsonl(documents: list[dict], output_path: Path) -> None:
    """Save documents as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")
    print(f"Saved {len(documents)} documents to {output_path}")


def save_parquet(documents: list[dict], output_path: Path) -> None:
    """Save documents as Parquet."""
    try:
        import pandas as pd
    except ImportError:
        print("pandas required for parquet output: pip install pandas pyarrow")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(documents)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(documents)} documents to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate sample data for NeMo Curator")
    parser.add_argument("--output-dir", type=str, default="sample_data", help="Output directory")
    parser.add_argument("--format", type=str, choices=["jsonl", "parquet"], default="jsonl")
    parser.add_argument("--num-docs", type=int, default=100, help="Number of documents")
    parser.add_argument("--good-ratio", type=float, default=0.7, help="Ratio of good documents")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    documents = generate_dataset(args.num_docs, args.good_ratio)

    output_dir = Path(args.output_dir)
    if args.format == "jsonl":
        save_jsonl(documents, output_dir / "input.jsonl")
    else:
        save_parquet(documents, output_dir / "input.parquet")

    # Print summary
    print(f"\nDataset summary:")
    print(f"  Total documents: {len(documents)}")
    print(f"  Expected to pass: ~{int(args.num_docs * args.good_ratio)}")
    print(f"  Expected to filter: ~{int(args.num_docs * (1 - args.good_ratio))}")


if __name__ == "__main__":
    main()
