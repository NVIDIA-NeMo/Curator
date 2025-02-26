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

import importlib
import os
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Tuple, Union

import dask.dataframe as dd
import pandas as pd
from dask import compute, delayed

from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import read_single_partition


class DocumentDownloader(ABC):
    """Abstract class for downloading remote data to disk"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def download(self, url):
        pass


class DocumentIterator(ABC):
    """
    Abstract iterator class for reading in raw records that have been
    downloaded to disk
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def iterate(self, file_path):
        pass


class DocumentExtractor(ABC):
    """Abstract class for extracting text from records read from disk"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def extract(self, content):
        pass


def import_downloader(downloader_path):
    module_path, downloader_name = downloader_path.rsplit(".", 1)
    downloader_module = importlib.import_module(module_path)
    downloader_class = getattr(downloader_module, downloader_name)
    if not issubclass(downloader_class, DocumentDownloader):
        raise ValueError(
            f"Input downloader {downloader_class.__name__} "
            "must be derived from DocumentDownloader defined in "
            "nemo_curator.download.docbuilder"
        )
    return downloader_class


def import_iterator(iterator_path):
    module_path, iterator_name = iterator_path.rsplit(".", 1)
    iterator_module = importlib.import_module(module_path)
    iterator_class = getattr(iterator_module, iterator_name)
    if not issubclass(iterator_class, DocumentIterator):
        raise ValueError(
            f"Input iterator {iterator_class.__name__} "
            "must be derived from DocumentIterator "
            "defined in nemo_curator.download.docbuilder"
        )
    return iterator_class


def import_extractor(extractor_path):
    module_path, extractor_name = extractor_path.rsplit(".", 1)
    extractor_module = importlib.import_module(module_path)
    extractor_class = getattr(extractor_module, extractor_name)
    if not issubclass(extractor_class, DocumentExtractor):
        raise ValueError(
            f"Input extractor {extractor_class.__name__} "
            "must be derived from DocumentExtractor defined "
            "in nemo_curator.download.docbuilder"
        )
    return extractor_class


def _download_and_extract_single_partition(
    paths: List[Tuple[str, str]],
    downloader: DocumentDownloader,
    iterator: DocumentIterator,
    extractor: DocumentExtractor,
    output_type: Literal["jsonl", "parquet"],
    keep_raw_download: bool,
    force_download: bool,
    input_meta: Union[str, dict] = None,
    filename_col: str = "file_name",
    record_limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Downloads a single partition from a URL and extracts its contents in-memory without writing
    the extracted output to disk. The provided output_path is used only to check if an extracted
    output already exists and, if so, skips re-downloading and extraction.

    Parameters:
        paths (Tuple[str, str]): A tuple (url, output_path) where 'url' is the source URL and
            'output_path' is the expected location of a previously extracted output.
        downloader (DocumentDownloader): An object to download the content from the URL.
        iterator (DocumentIterator): An object to iterate over records in the downloaded file.
        extractor (DocumentExtractor): An object to extract the desired content from each record.
        output_type (Literal["jsonl", "parquet"]): The output file format/extension. Must be either "jsonl" or "parquet". Defaults to "jsonl". This parameter is only used to verify whether an extracted output already exists.
        keep_raw_download (bool): If False, deletes the raw download file after extraction.
        force_download (bool): If False and output_path exists, skips downloading and extraction.
        input_meta (Union[str, dict], optional): Metadata describing the input file's structure.
        filename_col (str, optional): Name of the column to store the filename within the result DataFrame.
        record_limit (int, optional): Limit the number of records to extract from each file.
    Returns:
        pd.DataFrame: A DataFrame containing the extracted records.
    """
    url, output_path = paths

    # If an extracted output already exists and we're not forcing a download, load and return it.
    if os.path.exists(output_path) and not force_download:
        partition = read_single_partition(
            [output_path],
            backend="pandas",
            file_type=output_type,
            add_filename=filename_col,
        )
        return partition

    # Download the file and extract its records in memory.
    downloaded_file = downloader.download(url)
    records = []
    for item in iterator.iterate(downloaded_file):
        if record_limit is not None and len(records) >= record_limit:
            break
        record_meta, content = item
        extracted = extractor.extract(content)
        if extracted is not None:
            # Merge the extracted data and record metadata into one dictionary.
            line = {**extracted, **record_meta}
            records.append(line)

    partition = pd.DataFrame(records)
    # Add a filename column for consistency using the basename of the output_path.
    partition[filename_col] = os.path.basename(output_path)

    # Since we're not writing the extracted partition to disk, the output_path is not used here.
    # Clean up the raw downloaded file if it's not meant to be kept.
    if not keep_raw_download and os.path.exists(downloaded_file):
        os.remove(downloaded_file)

    return partition


def download_and_extract(
    urls: List[str],
    output_paths: List[str],
    downloader: DocumentDownloader,
    iterator: DocumentIterator,
    extractor: DocumentExtractor,
    output_format: dict,
    output_type: Literal["jsonl", "parquet"] = "jsonl",
    keep_raw_download: bool = False,
    force_download: bool = False,
    input_meta: Union[str, dict] = None,
    filename_col: str = "file_name",
    record_limit: Optional[int] = None,
) -> DocumentDataset:
    """
    Download files from the given URLs, extract their records, and
    construct a DocumentDataset.

    For each URL provided, this function downloads the corresponding
    file (unless an extracted output already exists and force_download is
    False), iterates over its records, extracts the desired content, and
    finally converts all records into a DocumentDataset.

    Args:
        urls (List[str]):
            A list of URLs from which to download dataset files.
        output_paths (List[str]):
            A list of file paths where the extracted outputs should be
            found. If a file already exists at a given path and force_download
            is False, that partition is skipped.
        downloader (DocumentDownloader):
            The downloader instance responsible for fetching files from
            the specified URLs.
        iterator (DocumentIterator):
            The iterator instance used to traverse the downloaded file
            and yield records.
        extractor (DocumentExtractor):
            The extractor instance used to obtain the desired content from
            each record.
        output_format (dict):
            A dictionary mapping column names to the data types for the
            extracted records.
        output_type (Literal["jsonl", "parquet"], optional):
            The output file format/extension. Must be either "jsonl" or "parquet".
            Defaults to "jsonl". This parameter is only used to verify whether
            an extracted output already exists.
        keep_raw_download (bool, optional):
            If True, the raw downloaded files are retained after extraction.
            Defaults to False.
        force_download (bool, optional):
            If False and an output file already exists at a given path, the
            download and extraction for that file are skipped.
            Defaults to False.
        input_meta (Union[str, dict], optional):
            Optional metadata describing the input file's schema.
            Defaults to None.
        filename_col (str, optional):
            The name for the column in the resulting dataset that records
            the basename of the output file. Defaults to "file_name".
        record_limit (int, optional): Limit the number of records to extract from each file.
            Defaults to None.
    Returns:
        DocumentDataset:
            A dataset composed of the records extracted from the downloaded
            files.
    """
    # Validate parameters
    if not urls:
        raise ValueError("No URLs were provided to download")
    if len(urls) != len(output_paths):
        raise ValueError(
            f"Different number of URLs and output_paths. {len(urls)} URLs vs {len(output_paths)} output_paths"
        )

    # Ensure consistent ordering of output_format keys.
    output_format = dict(sorted(output_format.items()))

    df = dd.from_map(
        _download_and_extract_single_partition,
        zip(urls, output_paths),
        downloader=downloader,
        iterator=iterator,
        extractor=extractor,
        output_type=output_type,
        keep_raw_download=keep_raw_download,
        force_download=force_download,
        enforce_metadata=False,
        input_meta=input_meta,
        filename_col=filename_col,
        record_limit=record_limit,
        meta=output_format,
    )

    return DocumentDataset(df)


def batch_download(urls: List[str], downloader: DocumentDownloader) -> List[str]:
    """
    Downloads all the urls using the downloader in parallel
    """
    delayed_downloads = [delayed(downloader.download)(url) for url in urls]

    return compute(*delayed_downloads)
