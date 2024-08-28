# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import os
from typing import Optional

import pandas as pd
from docbuilder import (
    ArxivDownloader,
    ArxivExtractor,
    ArxivIterator,
    GitHubDownloader,
    GitHubExtractor,
    GitHubIterator,
    WikitxtDownloader,
    WikitxtExtractor,
    WikitxtIterator,
)

from nemo_curator.download.doc_builder import download_and_extract


def download_wikipedia_sources(
    source_links_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """
    Downloads Wikipedia sources based on the provided source links file.

    Args:
        source_links_file: Path to the file containing the source links. If not provided,
            a default file path (./sources/wikipedia_urls.jsonl) will be used.
        output_dir: Directory where the downloaded sources will be saved. If not provided,
            a default directory path (./data/raw/wikipedia) will be used.
        limit: Maximum number of sources to download. If provided, only the first `limit`
            sources will be downloaded.

    Returns:
        str: The path to the output directory where the downloaded sources are saved.
    """

    if source_links_file is None:
        source_links_file = os.path.join(
            os.path.dirname(__file__), "sources", "wikipedia_urls.jsonl"
        )

    if not os.path.exists(source_links_file):
        raise FileNotFoundError(f"File '{source_links_file}' not found.")

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "data", "raw", "wikipedia")

    os.makedirs(output_dir, exist_ok=True)
    # Read the source links from the provided file
    urls = pd.read_json(path_or_buf=source_links_file, lines=True)
    urls = urls[0].tolist()

    if limit:
        urls = urls[:limit]
    output_format = {
        "text": str,
        "id": str,
        "file_extension": str,
        "file_type": str,
        "category": str,
        "line_count": int,
        "size_in_bytes": int,
        "path": str,
    }

    downloader = WikitxtDownloader(output_dir)
    iterator = WikitxtIterator()
    extractor = WikitxtExtractor()

    dataset = download_and_extract(
        urls=urls,
        output_paths=[os.path.join(output_dir, os.path.basename(url)) for url in urls],
        downloader=downloader,
        iterator=iterator,
        extractor=extractor,
        output_format=output_format,
    )
    # Force the computation of the dataset
    dataset.persist()
    return output_dir


def download_github_sources(
    source_links_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    limit: Optional[int] = None,
):
    """
    Downloads GitHub sources specified in a file and extracts them.

    Args:
        source_links_file: Path to the file containing the GitHub source links.
        output_dir: Directory where the downloaded and extracted files will be stored.
        limit: Maximum number of GitHub sources to download and extract.

    Raises:
        FileNotFoundError: If the source_links_file does not exist.

    Returns:
        str: Path to the output directory where the downloaded and extracted files are stored.
    """

    if source_links_file is None:
        source_links_file = os.path.join(
            os.path.dirname(__file__), "sources", "github_repos.jsonl"
        )

    if not os.path.exists(source_links_file):
        raise FileNotFoundError(f"File '{source_links_file}' not found.")

    urls = pd.read_json(path_or_buf=source_links_file, lines=True)
    urls = urls[0].tolist()

    if limit:
        urls = urls[:limit]

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "data", "raw", "github")

    os.makedirs(output_dir, exist_ok=True)
    output_jsonl_dir = os.path.join(output_dir, "jsonl")
    os.makedirs(output_jsonl_dir, exist_ok=True)

    downloader = GitHubDownloader(output_dir)
    iterator = GitHubIterator()
    extractor = GitHubExtractor()

    output_format = {
        "text": str,
        "id": str,
        "file_extension": str,
        "file_type": str,
        "category": str,
        "line_count": int,
        "size_in_bytes": int,
        "path": str,
    }

    dataset = download_and_extract(
        urls=urls,
        output_paths=[
            os.path.join(output_jsonl_dir, os.path.basename(url)) for url in urls
        ],
        downloader=downloader,
        iterator=iterator,
        extractor=extractor,
        output_format=output_format,
        keep_raw_download=True,
    )
    # Force the computation of the dataset
    dataset.persist()
    return output_jsonl_dir


def download_pdf_sources(
    source_links_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    limit: Optional[int] = None,
):
    """
    Downloads Arxiv Pdf sources specified in a file and extracts them.

    Args:
        source_links_file: Path to the file containing the Arxiv PDF source links.
        output_dir: Directory where the downloaded and extracted files will be stored.
        limit: Maximum number of pdf sources to download and extract.

    Raises:
        FileNotFoundError: If the source_links_file does not exist.

    Returns:
        str: Path to the output directory where the downloaded and extracted files are stored.
    """

    if source_links_file is None:
        source_links_file = os.path.join(
            os.path.dirname(__file__), "sources", "arxiv_urls.jsonl"
        )

    if not os.path.exists(source_links_file):
        raise FileNotFoundError(f"File '{source_links_file}' not found.")

    urls = pd.read_json(path_or_buf=source_links_file, lines=True)
    urls = urls[0].tolist()

    if limit:
        urls = urls[:limit]

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(__file__), "data", "raw", "arxiv_pdfs"
        )

    os.makedirs(output_dir, exist_ok=True)
    output_jsonl_dir = os.path.join(output_dir, "jsonl")
    os.makedirs(output_jsonl_dir, exist_ok=True)

    downloader = ArxivDownloader(output_dir)
    iterator = ArxivIterator()
    extractor = ArxivExtractor()

    output_format = {
        "text": str,
        "id": str,
        "file_extension": str,
        "file_type": str,
        "category": str,
        "line_count": int,
        "size_in_bytes": int,
        "path": str,
    }

    dataset = download_and_extract(
        urls=urls,
        output_paths=[
            os.path.join(output_jsonl_dir, os.path.basename(url)) for url in urls
        ],
        downloader=downloader,
        iterator=iterator,
        extractor=extractor,
        output_format=output_format,
        keep_raw_download=True,
    )
    # Force the computation of the dataset
    dataset.persist()
    return output_jsonl_dir
