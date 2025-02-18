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
import subprocess
import unicodedata
from abc import ABC, abstractmethod
from typing import Literal, Optional
from urllib.parse import urlparse

import justext
import lxml
import pycld2 as cld2
from charset_normalizer import detect
from resiliparse.extract.html2text import extract_plain_text
from warcio.archiveiterator import ArchiveIterator

from nemo_curator.datasets import DocumentDataset
from nemo_curator.download.doc_builder import (
    DocumentDownloader,
    DocumentExtractor,
    DocumentIterator,
    download_and_extract,
)
from nemo_curator.utils.download_utils import get_common_crawl_urls
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir


def decode_html(html_bytes):
    # Convert from bytes to text using utf-8 encoding
    try:
        return html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # If utf-8 fails, try to find a different encoding
        return try_decode_with_detected_encoding(html_bytes)


def try_decode_with_detected_encoding(html_bytes):
    detected_encoding = detect(html_bytes)["encoding"]
    bad_detection = not detected_encoding or detected_encoding == "utf-8"
    if bad_detection:
        return None
    try:
        return html_bytes.decode(detected_encoding)
    except:
        return None


def lang_detect(decoded_html):
    try:
        details = cld2.detect(decoded_html)[2]
    except Exception:
        # Remove control characters
        cleaned_html = "".join(
            i for i in decoded_html if unicodedata.category(i)[0] != "C"
        )
        details = cld2.detect(cleaned_html)[2]

    return details[0][0].upper()


class HTMLExtractorAlgorithm(ABC):
    @abstractmethod
    def extract_text(self, html, stop_words):
        pass


class JusTextExtractor(HTMLExtractorAlgorithm):
    def __init__(
        self,
        length_low=70,
        length_high=200,
        stopwords_low=0.30,
        stopwords_high=0.32,
        max_link_density=0.2,
        max_heading_distance=200,
        no_headings=False,
        logger=None,
    ):
        """
        Initialize the jusText text extraction algorithm with specified parameters.

        Args:
            length_low: Minimum length of text to be considered for extraction.
            length_high: Maximum length of text to be considered for extraction.
            stopwords_low: Minimum proportion of stopwords in the text to be considered for extraction.
            stopwords_high: Maximum proportion of stopwords in the text to be considered for extraction.
            max_link_density: Maximum allowed link density in the text.
            max_heading_distance: Maximum distance from a heading to consider text for extraction.
            no_headings: If True, text extraction will ignore headings.
            logger: Optional logger instance for logging messages.

        """
        self.length_low = length_low
        self.length_high = length_high
        self.stopwords_low = stopwords_low
        self.stopwords_high = stopwords_high
        self.max_link_density = max_link_density
        self.max_heading_distance = max_heading_distance
        self.no_headings = no_headings
        self.logger = logger

    def extract_text(self, html, stop_words):
        # Segment the HTML into paragraphs
        try:
            # Form the DOM tree
            dom = justext.core.html_to_dom(html)
            cleaned_dom = justext.core.preprocessor(dom)
            # Get the paragraphs from the DOM
            handler = justext.core.ParagraphMaker()
            lxml.sax.saxify(cleaned_dom, handler)
        except (lxml.etree.ParserError, ValueError, Exception):
            # Return nothing when we cannot segment the document
            if self.logger is not None:
                self.logger.info("Could not segment paragaphs in the document")
            return
        paragraphs = handler.paragraphs

        # Context free classification
        justext.core.classify_paragraphs(
            paragraphs,
            stop_words,
            self.length_low,
            self.length_high,
            self.stopwords_low,
            self.stopwords_high,
            self.max_link_density,
            self.no_headings,
        )

        # Copy the context free class to the class_style
        # This handles the headings as described in the
        # documentation
        for paragraph in paragraphs:
            paragraph.class_type = paragraph.cf_class

        # Context sensitive classification
        justext.core.revise_paragraph_classification(
            paragraphs,
            self.max_heading_distance,
        )

        return [p.text for p in paragraphs if not p.is_boilerplate]


class ResiliparseExtractor(HTMLExtractorAlgorithm):
    def __init__(
        self,
        required_stopword_density=0.32,
        main_content=True,
        alt_texts=False,
    ):
        """
        Initialize the Resiliparse text extraction algorithm with specified parameters.

        Args:
            required_stopword_density: Proportion of stopwords required preserve an extracted paragraph.
                Studies on stopword lists and their distribution in various text corpora often
                suggest that around 30-40% of a typical English text consists of stopwords.
            main_content: Whether to apply simple heuristics for extracting only "main-content" elements.
            alt_texts: Whether to preserve alternative text descriptions (e.g., for images).

        """
        self.required_stopword_density = required_stopword_density
        self.main_content = main_content
        self.alt_texts = alt_texts

    def extract_text(self, html, stop_words):
        text = extract_plain_text(
            html, main_content=self.main_content, alt_texts=self.alt_texts
        )

        paragraphs = list(filter(None, text.split("\n")))
        result = []
        for paragraph in paragraphs:
            words = paragraph.split()
            length = len(words)
            if length == 0:
                continue
            stopwords = [word for word in words if word in stop_words]
            stopword_density = len(stopwords) / length

            if stopword_density >= self.required_stopword_density:
                result.append(paragraph)

        if len(result) == 0:
            return None
        return result


def get_stop_list_dict(languages=[]):

    # Name mapping for language names from CLD2 (values)
    # and jusText (keys)
    lang_map = {
        "Haitian": "HAITIAN_CREOLE",
        "Norwegian_Bokmal": "NORWEGIAN",
        "Norwegian_Nynorsk": "NORWEGIAN_N",
        "Waray_Waray": "WARAY_PHILIPPINES",
    }
    if len(languages) == 0:
        languages = justext.get_stoplists()
        # Remove latin as it yields a lot of low quality documents
        languages_no_latin = list(languages)
        languages_no_latin.remove("Latin")
        languages = frozenset(languages_no_latin)

    stop_list_dict = {}
    for language in languages:
        if language in lang_map:
            lang_key = lang_map[language]
        else:
            lang_key = language.upper()
        stop_list_dict[lang_key] = justext.get_stoplist(language)

    # List obtained from https://github.com/stopwords-iso/stopwords-th
    from .thai_stopwords import thai_stopwords

    stop_list_dict["THAI"] = thai_stopwords

    return stop_list_dict


def get_all_stop_words():
    stop_words = set()
    for language in justext.get_stoplists():
        stop_words.update(justext.get_stoplist(language))

    return frozenset(stop_words)


class CommonCrawlWARCDownloader(DocumentDownloader):
    """
    Downloads WARC files from the Common Crawl
    """

    def __init__(self, download_dir, aws=False, verbose=False):
        """
        Creates a downloader

        Args:
          download_dir: Path to store raw compressed WARC files
          aws: If True, uses the s5cmd command to download from the Common Crawl's S3 bucket.
            If False, uses wget.
          verbose: If True, logs stdout and stderr of the download command (s5cmd/wget)
        """
        super().__init__()
        self._download_dir = download_dir
        self._aws = aws
        self._verbose = verbose

    def download(self, url):
        # Download each URL to the directory
        urlpath = urlparse(url).path[1:]
        output_name = urlpath.replace("/", "-")
        output_file = os.path.join(self._download_dir, output_name)
        if os.path.exists(output_file):
            print(f"WARC file: {output_file} exists. Not downloading")
        else:
            print(f"Downloading {url} and writing to {output_file}")
            # Download with either wget or s5cmd (aws)
            if self._aws:
                s3path = os.path.join("s3://commoncrawl/", urlpath)
                cmd = ["s5cmd", "cp", s3path, output_file]
            else:
                cmd = ["wget", url, "-O", output_file]
            if self._verbose:
                stdout, stderr = None, None
            else:
                stdout, stderr = subprocess.DEVNULL, subprocess.DEVNULL
            p = subprocess.run(
                cmd,
                stdout=stdout,
                stderr=stderr,
            )
            if p.returncode != 0:
                print(f"Failed to download {url} to {output_file}")

        return output_file


class CommonCrawlWARCDownloaderExtractOnly(DocumentDownloader):
    """
    A 'dummy' downloader that simply puts pre-downloaded
    files on the queue
    """

    def __init__(self, aws=False, verbose=False):
        super().__init__()

    def download(self, url):
        print(f"Putting WARC file {url} on the queue for extraction")
        return url


class CommonCrawlWARCIterator(DocumentIterator):

    def __init__(self, log_frequency=1000):
        super().__init__()
        self._counter = 0
        self._log_frequency = log_frequency

    def iterate(self, file_path):
        # Loop over all records in the current WARC
        self._counter = 0
        bname = os.path.split(file_path)[-1]
        with open(file_path, "rb") as file_pointer:
            ai = ArchiveIterator(file_pointer, arc2warc=True)
            for k, rec in enumerate(ai):
                # Get the response from the crawl
                if rec.rec_type == "response":
                    if self._counter > 0 and self._counter % self._log_frequency == 0:
                        print(f"Extracted {self._counter} records in WARC")
                    self._counter += 1
                    content = rec.content_stream().read()
                    warc_id = rec.rec_headers.get_header("WARC-Record-ID")[10:-1]
                    url = rec.rec_headers.get_header("WARC-Target-URI")
                    meta = {
                        "url": url,
                        "warc_id": warc_id,
                        "source_id": f"{bname}",
                    }
                    yield meta, content


class CommonCrawlWARCExtractor(DocumentExtractor):

    def __init__(self, algorithm=JusTextExtractor()):
        self._stop_lists = get_stop_list_dict()
        self.algorithm = algorithm
        super().__init__()

    def extract(self, content):
        html = decode_html(content)
        if html is not None:
            # Language detection and HTML extraction
            lang = lang_detect(html)
            text = None
            if lang in self._stop_lists:
                text = self.algorithm.extract_text(html, self._stop_lists[lang])
            if text is not None:
                if len(text) > 0:
                    text = "\n\n".join(text)
                    meta = {"language": lang, "text": text}
                    return meta
                else:
                    return None


def download_common_crawl(
    output_path: str,
    start_snapshot: str,
    end_snapshot: str,
    output_type: Literal["jsonl", "parquet"] = "jsonl",
    algorithm=JusTextExtractor(),
    news: bool = False,
    aws: bool = False,
    raw_download_dir: Optional[str] = None,
    keep_raw_download: bool = False,
    force_download: bool = False,
    url_limit: Optional[int] = None,
    record_limit: Optional[int] = None,
) -> DocumentDataset:
    """
    Downloads Common Crawl WARC snapshots and extracts text content using a specified extraction algorithm.

    Args:
      output_path (str): The root directory used for managing download and extraction.
          • Raw WARC files are stored in a "downloads" subdirectory under this path.
          • This path is also checked for existing extraction results; if found, extraction can be skipped.
          • Note: This function returns a DocumentDataset, and writing the extracted data to disk is the caller's responsibility.
      start_snapshot (str): Identifier for the earliest snapshot to process.
          • For CC-MAIN datasets, use the format 'YYYY-WeekNumber' (e.g., '2020-50' or '2021-04').
          • For CC-NEWS datasets (when news=True), use the 'YYYY-MM' (Year-Month) format.
      end_snapshot (str): Identifier for the latest snapshot to process, which must be chronologically after start_snapshot.
      output_type (Literal["jsonl", "parquet"]): The file format for the extracted output. Must be either "jsonl" or "parquet".
          • This is not used for the output file, but is used to check if an extracted output already exists.
      algorithm: The text extraction algorithm instance (e.g., JusTextExtractor or ResiliparseExtractor) to use for HTML processing.
      news (bool): When True, indicates that URLs should be retrieved from the CC-NEWS dataset.
          • This also means snapshot identifiers should follow the 'YYYY-MM' format.
      aws (bool): If True, downloads are sourced from Common Crawl's S3 bucket using s5cmd;
          • If False, wget is used to fetch the files via HTTPS.
      raw_download_dir: Optional; the directory to temporarily store raw WARC files.
          • If not provided, defaults to a "downloads" folder within output_path.
      keep_raw_download (bool): If True, retains the downloaded raw WARC files after extraction.
          • If False, these raw files may be removed following extraction.
      force_download (bool): If False, skips re-downloading or re-extracting snapshots if outputs already exist in output_path.
      url_limit: Optional; the maximum number of WARC files to download from the snapshot range.
          • If None, all available files within the specified snapshots are downloaded.
      record_limit: Optional; the maximum number of records to extract from each WARC file.
          • If None, all available records are extracted.
    """
    common_crawl_urls = get_common_crawl_urls(
        starting_snapshot=start_snapshot, ending_snapshot=end_snapshot, news=news
    )

    if len(common_crawl_urls) == 0:
        raise ValueError(
            f"No Common Crawl download urls found between {start_snapshot} and {end_snapshot}. "
            "Ensure that a valid Common Crawl snapshot (https://data.commoncrawl.org/) is "
            "within the range provided."
        )

    if url_limit:
        common_crawl_urls = common_crawl_urls[:url_limit]
    output_paths = list(
        map(
            lambda url: os.path.join(
                output_path, url.split("/")[-1] + f".{output_type}"
            ),
            common_crawl_urls,
        )
    )

    if not raw_download_dir:
        raw_download_dir = os.path.join(output_path, "downloads")
    expand_outdir_and_mkdir(raw_download_dir)
    downloader = CommonCrawlWARCDownloader(raw_download_dir, aws=aws)
    iterator = CommonCrawlWARCIterator()
    extractor = CommonCrawlWARCExtractor(algorithm=algorithm)

    output_format = {
        "text": str,
        "language": str,
        "url": str,
        "warc_id": str,
        "source_id": str,
        "file_name": str,
    }
    dataset = download_and_extract(
        common_crawl_urls,
        output_paths,
        downloader,
        iterator,
        extractor,
        output_format,
        output_type=output_type,
        keep_raw_download=keep_raw_download,
        force_download=force_download,
        filename_col="file_name",
        record_limit=record_limit,
    )

    return dataset
