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
from urllib.parse import urlparse

import justext
import lxml
import pycld2 as cld2
from charset_normalizer import detect
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


def extract_text(
    html,
    stop_words,
    length_low=70,
    length_high=200,
    stopwords_low=0.30,
    stopwords_high=0.32,
    max_link_density=0.2,
    max_heading_distance=200,
    no_headings=False,
    logger=None,
):
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
        if logger is not None:
            logger.info("Could not segment paragaphs in the document")
        return
    paragraphs = handler.paragraphs

    # Context free classification
    justext.core.classify_paragraphs(
        paragraphs,
        stop_words,
        length_low,
        length_high,
        stopwords_low,
        stopwords_high,
        max_link_density,
        no_headings,
    )

    # Copy the context free class to the class_style
    # This handles the headings as described in the
    # documentation
    for paragraph in paragraphs:
        paragraph.class_type = paragraph.cf_class

    # Context sensitive classification
    justext.core.revise_paragraph_classification(
        paragraphs,
        max_heading_distance,
    )

    return [p.text for p in paragraphs if not p.is_boilerplate]


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

    def __init__(self):
        self._stop_lists = get_stop_list_dict()
        super().__init__()

    def extract(self, content):
        html = decode_html(content)
        if html is not None:
            # Language detection and HTML extraction
            lang = lang_detect(html)
            text = None
            if lang in self._stop_lists:
                text = extract_text(html, self._stop_lists[lang])
            if text is not None:
                if len(text) > 0:
                    text = "\n\n".join(text)
                    meta = {"language": lang}
                    return meta, text
                else:
                    return None, None


def download_common_crawl(
    output_path: str,
    start_snapshot: str,
    end_snapshot: str,
    output_type: str = "jsonl",
    news=False,
    aws=False,
    raw_download_dir=None,
    keep_raw_download=False,
    force_download=False,
    url_limit=None,
) -> DocumentDataset:
    """
    Downloads Common Crawl WARC snapshots and extracts them using jusText

    Args:
      output_path: The path to the root directory of the files
      start_snapshot: The first common crawl snapshot to include. Snapshots must be
        specified by YYYY-WeekNumber (e.g., '2020-50' or '2021-04'). For the CC-NEWS dataset,
        (specified with news=True flag) this changes to Year-Month (YYYY-MM).
      end_snapshot: The last common crawl snapshot to include. Must be chronologically
        after the starting snapshot.
      output_type: The file type to save the data as.
      news: If True, gets WARC URLs for the CC-NEWS dataset instead of the CC-MAIN datasets.
        Also assumes that the format for the start and end snapshots is 'YYYY-MM' (Year-Month).
      aws: Whether to download from Common Crawl's S3 bucket. If True, uses s5cmd to download.
        If False, uses wget.
      raw_download_dir: Path to store the raw download files for intermediate processing.
        If None, they are stored in a folder named "downloads" under output_path.
      keep_raw_download: If True, keeps the compressed WARC files that have not been extracted.
      force_download: If False, will skip processing all files in output_paths that already exist and
        directly read from them instead.
      url_limit: The maximum number of raw files to download from the snapshot. If None, all
        files from the range of snapshots are downloaded.
    """
    common_crawl_urls = get_common_crawl_urls(
        starting_snapshot=start_snapshot, ending_snapshot=end_snapshot, news=news
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
    extractor = CommonCrawlWARCExtractor()

    output_format = {
        "text": str,
        "language": str,
        "url": str,
        "warc_id": str,
        "source_id": str,
        "filename": str,
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
    )

    return dataset
