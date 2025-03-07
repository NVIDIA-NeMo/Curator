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


import os
import subprocess
import unicodedata
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Literal, Optional
from urllib.parse import urlparse

import justext
import lxml
import pycld2 as cld2
from charset_normalizer import detect
from resiliparse.extract.html2text import extract_plain_text
from trafilatura import extract as extract_with_trafilatura
from trafilatura.settings import DEFAULT_CONFIG as TRAFILATURA_DEFAULT_CONFIG
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

NON_SPACED_LANGUAGES = ["THAI", "CHINESE", "JAPANESE", "KOREAN"]


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
    def extract_text(self, html, stop_words, language):
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
        is_boilerplate=None,
        logger=None,
    ):
        """
        Initialize the jusText text extraction algorithm with specified parameters.

        jusText is a tool for removing boilerplate content, such as navigation links, headers, and footers from HTML pages.
        It is designed to preserve mainly text containing full sentences and it is therefore well suited for creating linguistic resources such as Web corpora.
        The key idea is that long blocks can often be classified with high confidence, while shorter blocks require context-based adjustments.

        Here is an overview of the jusText algorithm:
            • Segmentation: The document is split into textual blocks based on HTML tags that typically define separate sections (e.g., <div>, <p>, <table>).
            • Preprocessing: Contents of <header>, <style>, and <script> tags are removed.
                Certain elements (e.g., <select>, copyright symbols) are immediately classified as boilerplate.
            • Context-Free Classification: Each block is classified as:
                - Bad (boilerplate) if it has high link density.
                - Short if it is too small to be classified reliably.
                - Near-Good if it has a moderate density of stopwords.
                - Good (main content) if it is long and contains many stopwords.
            • Context-Sensitive Classification: Blocks that were classified as short or near-good are reclassified based on surrounding blocks.
                The assumption is that main content clusters together, as does boilerplate.
            • Headings Processing: Header elements (e.g., <h1>, <h2>) are treated separately to ensure useful headings are preserved.
                Short headers near good content may be reclassified as near-good or good.

        Please refer to the jusText documentation for more details: https://corpus.tools/wiki/Justext/Algorithm

        Args:
            length_low: Minimum length of text to be considered for extraction.
            length_high: Maximum length of text to be considered for extraction.
            stopwords_low: Minimum proportion of stopwords in the text to be considered for extraction.
            stopwords_high: Maximum proportion of stopwords in the text to be considered for extraction.
            max_link_density: Maximum allowed link density in the text.
            max_heading_distance: Maximum distance from a heading to consider text for extraction.
            no_headings: If True, text extraction will ignore headings.
            is_boilerplate: If True, text extraction will ignore boilerplate content.
                Default is True for space-separated languages and False for non-space-separated languages
                (Thai, Chinese, Japanese, and Korean).
            logger: Optional logger instance for logging messages.

        """
        self.length_low = length_low
        self.length_high = length_high
        self.stopwords_low = stopwords_low
        self.stopwords_high = stopwords_high
        self.max_link_density = max_link_density
        self.max_heading_distance = max_heading_distance
        self.no_headings = no_headings
        self.is_boilerplate = is_boilerplate
        self.logger = logger

    def extract_text(self, html, stop_words, language):
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

        if self.is_boilerplate is None:
            if language in NON_SPACED_LANGUAGES:
                warnings.warn("Disabling is_boilerplate check for jusText extraction.")
                is_boilerplate = False
            else:
                is_boilerplate = True

        else:
            is_boilerplate = self.is_boilerplate

        if is_boilerplate:
            return [p.text for p in paragraphs if not p.is_boilerplate]

        else:
            return [p.text for p in paragraphs]


class ResiliparseExtractor(HTMLExtractorAlgorithm):
    def __init__(
        self,
        required_stopword_density=0.32,
        main_content=True,
        alt_texts=False,
    ):
        """
        Initialize the Resiliparse text extraction algorithm with specified parameters.

        The Resiliparse algorithm extracts structural or semantic information from noisy raw web data for further processing,
        such as (main) content extraction / boilerplate removal, schema extraction, general web data cleansing, and more.

        It is implemented via the `extract_plain_text` function in the `resiliparse.extract.html2text` module.
        Resiliparse HTML2Text is a very fast and rule-based plain text extractor for HTML pages which uses the Resiliparse DOM parser.
        The `extract_plain_text` function extracts all visible text nodes inside the HTML document's <body>.
        Only <script>, <style> and a few other (generally) invisible elements are skipped and very basic ASCII formatting is applied.

        Please refer to the Resiliparse documentation for more details: https://resiliparse.chatnoir.eu/en/latest/man/extract/html2text.html

        NeMo Curator has added a stopword density filter to the Resiliparse extraction process, which requires that a paragraph contains a certain proportion of stopwords.

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

    def extract_text(self, html, stop_words, language):
        text = extract_plain_text(
            html, main_content=self.main_content, alt_texts=self.alt_texts
        )

        paragraphs = list(filter(None, text.split("\n")))

        if language in NON_SPACED_LANGUAGES:
            warnings.warn(
                "stopword_density is ignored for non-space-separated languages."
            )
            result = paragraphs
        else:
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

        return result


class TrafilaturaExtractor(HTMLExtractorAlgorithm):
    def __init__(
        self,
        required_stopword_density=0.32,
        min_extracted_size=250,
        min_extracted_comm_size=1,
        min_output_size=1,
        min_output_comm_size=1,
        max_tree_size=None,
        min_duplcheck_size=100,
        max_repetitions=2,
        **extract_kwargs,
    ):
        """
        Initialize the Trafilatura text extraction algorithm with specified parameters.

        The Trafilatura extraction process combines readability-lxml and jusText as fallbacks to ensure robustness.
        Trafilatura's own algorithm follows a cascade of rule-based filters and content heuristics:
            • Content Delimitation: Uses XPath expressions to exclude unwanted HTML elements (e.g., navigation bars) and focus on relevant content (e.g., article body).
                Extracted HTML nodes are analyzed for relevance based on element type, text length, and link density.
            • Fallback Mechanism: If extraction seems faulty, alternative algorithms are run as backups.
                These use heuristics like line length, text-to-markup ratio, and HTML depth to improve extraction.
                Outputs are compared, prioritizing longer extractions with fewer impurities.
            • Baseline Extraction: If all else fails, it searches for text elements that might have been missed, discarding irrelevant content.

        The system balances precision and recall, extracting main text, comments, and metadata (title, site name, author, date, categories, tags).

        Please refer to the Trafilatura documentation for more details:
            https://trafilatura.readthedocs.io/en/latest/ and https://aclanthology.org/2021.acl-demo.15/

        NeMo Curator has added a stopword density filter to the Trafilatura extraction process, which requires that a paragraph contains a certain proportion of stopwords.

        Args:
            required_stopword_density: Proportion of stopwords required preserve an extracted paragraph.
                Studies on stopword lists and their distribution in various text corpora often
                suggest that around 30-40% of a typical English text consists of stopwords.
            min_extracted_size: Acceptable size in characters (used to trigger fallbacks).
                Defaults to 250. See Trafilatura documentation: https://trafilatura.readthedocs.io/en/latest/settings.html.
            min_extracted_comm_size: Works the same as min_output_comm_size for comment extraction.
                Defaults to 1. See Trafilatura documentation: https://trafilatura.readthedocs.io/en/latest/settings.html.
            min_output_size: Absolute acceptable minimum for main text output.
                Defaults to 1. See Trafilatura documentation: https://trafilatura.readthedocs.io/en/latest/settings.html.
            min_output_comm_size: Works the same as min_output_comm_size for comment extraction.
                Defaults to 1. See Trafilatura documentation: https://trafilatura.readthedocs.io/en/latest/settings.html.
            max_tree_size: Used to discard documents with too many elements. Defaults to None.
            min_duplcheck_size: Minimum size in characters to run deduplication on.
                Defaults to 100. See Trafilatura documentation: https://trafilatura.readthedocs.io/en/latest/settings.html.
            max_repetitions: Maximum number of duplicates allowed.
                Defaults to 2. See Trafilatura documentation: https://trafilatura.readthedocs.io/en/latest/settings.html.
            extract_kwargs: Additional keyword arguments for the Trafilatura extract function.
                See API documentation https://trafilatura.readthedocs.io/en/latest/corefunctions.html#extract
                for list of possible parameters.
                All arguments are set to their default values, except for deduplicate (bool) which is set to True.

        """
        self.required_stopword_density = required_stopword_density
        self.min_extracted_size = min_extracted_size
        self.min_extracted_comm_size = min_extracted_comm_size
        self.min_output_size = min_output_size
        self.min_output_comm_size = min_output_comm_size
        self.max_tree_size = max_tree_size
        self.min_duplcheck_size = min_duplcheck_size
        self.max_repetitions = max_repetitions
        self.extract_kwargs = extract_kwargs

    def extract_text(self, html, stop_words, language):
        trafilatura_config = deepcopy(TRAFILATURA_DEFAULT_CONFIG)
        trafilatura_config["DEFAULT"]["MIN_EXTRACTED_SIZE"] = str(
            self.min_extracted_size
        )
        trafilatura_config["DEFAULT"]["MIN_EXTRACTED_COMM_SIZE"] = str(
            self.min_extracted_comm_size
        )
        trafilatura_config["DEFAULT"]["MIN_OUTPUT_SIZE"] = str(self.min_output_size)
        trafilatura_config["DEFAULT"]["MIN_OUTPUT_COMM_SIZE"] = str(
            self.min_output_comm_size
        )
        if self.max_tree_size:
            trafilatura_config["DEFAULT"]["MAX_TREE_SIZE"] = str(self.max_tree_size)
        trafilatura_config["DEFAULT"]["MIN_DUPLCHECK_SIZE"] = str(
            self.min_duplcheck_size
        )
        trafilatura_config["DEFAULT"]["MAX_REPETITIONS"] = str(self.max_repetitions)

        # Recommended to set deduplicate=True
        self.extract_kwargs.setdefault("deduplicate", True)

        text = extract_with_trafilatura(
            html, config=trafilatura_config, **self.extract_kwargs
        )

        if text is not None:
            paragraphs = list(filter(None, text.split("\n")))

            if language in NON_SPACED_LANGUAGES:
                warnings.warn(
                    "stopword_density is ignored for non-space-separated languages."
                )
                result = paragraphs

            else:
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

        else:
            return None

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

    # List obtained from https://github.com/stopwords-iso/stopwords-ja
    from .ja_stopwords import ja_stopwords

    # List obtained from https://github.com/stopwords-iso/stopwords-th
    from .th_stopwords import th_stopwords

    # List obtained from https://github.com/stopwords-iso/stopwords-zh
    from .zh_stopwords import zh_stopwords

    custom_stopwords = {
        "THAI": th_stopwords,
        "CHINESE": zh_stopwords,
        "JAPANESE": ja_stopwords,
    }

    if len(languages) == 0:
        languages = justext.get_stoplists()

        # Remove Latin as it yields a lot of low quality documents
        languages = list(languages)
        languages.remove("Latin")

        # Manually add Thai, Chinese, and Japanese
        languages.append("THAI")
        languages.append("CHINESE")
        languages.append("JAPANESE")

        languages = frozenset(languages)

    stop_list_dict = {}
    for language in languages:
        if language in lang_map:
            lang_key = lang_map[language]
        else:
            lang_key = language.upper()

        if lang_key in custom_stopwords:
            stop_list_dict[lang_key] = custom_stopwords[lang_key]
        else:
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

    def __init__(self, algorithm=JusTextExtractor(), stop_lists=None):
        if stop_lists is not None:
            self._stop_lists = stop_lists
        else:
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
                text = self.algorithm.extract_text(html, self._stop_lists[lang], lang)
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
    stop_lists=None,
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
      algorithm: The text extraction algorithm instance to use for HTML processing.
          • This can be a JusTextExtractor (default), ResiliparseExtractor, or TrafilaturaExtractor object.
      stop_lists: A dictionary stop lists, where the keys are languages (e.g., "ENGLISH")
          and the values are Python frozensets denoting the list of stop words for that language.
          If None, it defaults to jusText's stop lists: https://github.com/miso-belica/jusText/tree/main/justext/stoplists,
          with added Thai, Chinese, and Japanese support.
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
    extractor = CommonCrawlWARCExtractor(algorithm=algorithm, stop_lists=stop_lists)

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
