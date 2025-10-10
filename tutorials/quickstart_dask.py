"""
Dask-based Sentiment Analysis Pipeline

Standalone implementation that processes sample sentences through:
1. Data Generation: Create sample sentences
2. Word Count: Add word count column
3. Sentiment Analysis: GPU-based sentiment classification using HuggingFace transformers

This version uses Dask for distributed processing instead of Ray/NeMo Curator.
"""

import random
import warnings

import dask.dataframe as dd
import huggingface_hub
import pandas as pd
import torch
from dask.distributed import Client
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Sample sentences for the demo
SAMPLE_SENTENCES = ["I love this product", "I hate this product", "I'm neutral about this product"]

# Constants for sentiment analysis
NEGATIVE_THRESHOLD = 0.5
POSITIVE_THRESHOLD = 0.5


def generate_sample_data(num_sentences_per_partition: int, num_partitions: int) -> pd.DataFrame:
    """
    Generate sample data with sentences.

    Args:
        num_sentences_per_partition: Number of sentences per partition
        num_partitions: Number of partitions to create

    Returns:
        Combined DataFrame with all sentences
    """
    all_data = []

    for partition_id in range(num_partitions):
        # Sample sentences for this partition
        sampled_sentences = [
            random.choice(SAMPLE_SENTENCES)  # noqa: S311
            for _ in range(num_sentences_per_partition)
        ]

        partition_data = pd.DataFrame(
            {"sentence": sampled_sentences, "partition_id": partition_id, "sentence_id": range(len(sampled_sentences))}
        )
        all_data.append(partition_data)

    return pd.concat(all_data, ignore_index=True)


def add_word_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add word count column to the dataframe.

    Args:
        df: DataFrame with 'sentence' column

    Returns:
        DataFrame with added 'word_count' column
    """
    df = df.copy()
    df["word_count"] = df["sentence"].str.split().str.len()
    return df


class SentimentAnalyzer:
    """
    GPU-based sentiment analyzer using HuggingFace transformers.
    """

    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None

    def setup(self) -> None:
        """Initialize model and tokenizer."""
        logger.info(f"Loading sentiment model: {self.model_name}")

        # Download model if not cached
        try:
            huggingface_hub.snapshot_download(
                repo_id=self.model_name,
                local_files_only=False,
                resume_download=True,
            )
        except Exception as download_error:  # noqa: BLE001
            logger.warning(f"Could not download model: {download_error}, trying to load anyway...")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        logger.info(f"Model loaded on device: {self.device}")

    def predict_sentiment_batch(self, sentences: list[str]) -> list[str]:
        """
        Predict sentiment for a batch of sentences.

        Args:
            sentences: List of sentences to analyze

        Returns:
            List of sentiment labels
        """
        if not self.model or not self.tokenizer:
            error_msg = "Model not initialized. Call setup() first."
            raise RuntimeError(error_msg)

        if not sentences:
            return []

        # Tokenize sentences
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Convert to sentiment labels
        sentiment_scores = predictions.cpu().numpy()
        sentiment_labels = []

        for scores in sentiment_scores:
            # Assuming 0=negative, 1=neutral, 2=positive
            if scores[0] > NEGATIVE_THRESHOLD:
                sentiment_labels.append("negative")
            elif scores[2] > POSITIVE_THRESHOLD:
                sentiment_labels.append("positive")
            else:
                sentiment_labels.append("neutral")

        return sentiment_labels


def process_sentiment_partition(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Process sentiment for a partition of data.

    Args:
        df: DataFrame partition with sentences
        model_name: HuggingFace model name

    Returns:
        DataFrame with added sentiment column
    """
    if df.empty:
        df["sentiment"] = []
        return df

    # Initialize analyzer for this worker
    analyzer = SentimentAnalyzer(model_name)
    analyzer.setup()

    # Get sentences
    sentences = df["sentence"].tolist()

    # Predict sentiments
    sentiments = analyzer.predict_sentiment_batch(sentences)

    # Add to dataframe
    df = df.copy()
    df["sentiment"] = sentiments

    return df


def run_sentiment_analysis_pipeline(
    num_sentences_per_partition: int = 50,
    num_partitions: int = 10,
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    scheduler_address: str | None = None,
) -> pd.DataFrame:
    """
    Run the complete sentiment analysis pipeline using Dask.

    Args:
        num_sentences_per_partition: Number of sentences per partition
        num_partitions: Number of partitions to create
        model_name: HuggingFace model name for sentiment analysis
        scheduler_address: Dask scheduler address (None for local)

    Returns:
        Final processed DataFrame
    """

    # Initialize Dask client
    client = Client(scheduler_address) if scheduler_address else Client()  # Local cluster

    logger.info(f"Dask client initialized: {client}")

    try:
        # Step 1: Generate sample data
        logger.info("Step 1: Generating sample data...")
        raw_data = generate_sample_data(num_sentences_per_partition, num_partitions)
        logger.info(f"Generated {len(raw_data)} sentences across {num_partitions} partitions")

        # Step 2: Create Dask DataFrame and add word counts
        logger.info("Step 2: Adding word counts...")
        ddf = dd.from_pandas(raw_data, npartitions=num_partitions)
        ddf_with_counts = ddf.map_partitions(add_word_count)

        # Step 3: Add sentiment analysis
        logger.info("Step 3: Running sentiment analysis...")
        # Define output metadata for map_partitions
        meta = ddf_with_counts._meta.copy()
        meta["sentiment"] = pd.Series(dtype=str)
        ddf_with_sentiment = ddf_with_counts.map_partitions(process_sentiment_partition, model_name, meta=meta)

        # Compute final results
        logger.info("Computing results...")
        final_df = ddf_with_sentiment.compute()

        logger.info("Pipeline completed successfully!")
        return final_df

    finally:
        client.close()


def print_sample_results(df: pd.DataFrame, num_samples: int = 10) -> None:
    """Print sample results from the processed DataFrame."""
    print(f"\nSample Results (showing {num_samples} out of {len(df)} total):")
    print("=" * 80)

    sample_df = df.head(num_samples)
    for _, row in sample_df.iterrows():
        print(f"Sentence: {row['sentence']}")
        print(f"Words: {row['word_count']}, Sentiment: {row['sentiment']}")
        print("-" * 40)


def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print summary statistics of the results."""
    print("\nSummary Statistics:")
    print("=" * 50)
    print(f"Total sentences processed: {len(df)}")
    print(f"Average word count: {df['word_count'].mean():.2f}")

    # Sentiment distribution
    sentiment_counts = df["sentiment"].value_counts()
    print("\nSentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {sentiment.title()}: {count} ({percentage:.1f}%)")


def main() -> None:
    """Main function to run the sentiment analysis pipeline."""

    # Configuration
    config = {
        "num_sentences_per_partition": 20,
        "num_partitions": 5,
        "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "scheduler_address": None,  # Use None for local, or specify scheduler address
    }

    print("Dask Sentiment Analysis Pipeline")
    print("=" * 50)
    print(f"Configuration: {config}")
    print()

    # Run pipeline
    try:
        results = run_sentiment_analysis_pipeline(**config)

        # Display results
        print_sample_results(results, num_samples=10)
        print_summary_statistics(results)

        # Save results (optional)
        output_file = "sentiment_analysis_results.csv"
        results.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
