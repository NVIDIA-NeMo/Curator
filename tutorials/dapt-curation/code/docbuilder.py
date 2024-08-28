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

import gzip
import os
import re
from typing import Set, Tuple
from zipfile import ZipFile, ZipInfo

import arxiv as arxiv
import cchardet as chardet
import requests
from bs4 import BeautifulSoup
from unstructured.partition.auto import partition

from nemo_curator.download.doc_builder import (
    DocumentDownloader,
    DocumentExtractor,
    DocumentIterator,
)


class WikitxtDownloader(DocumentDownloader):
    """
    A class for downloading data from wiki urls.
    """

    def __init__(self, download_dir: str):
        """
        Initializes the DocBuilder object.

        Args:
            download_dir: The root directory for wiki URLs
        """
        super().__init__()

        if not os.path.isdir(download_dir):
            os.makedirs(download_dir)

        self._download_dir = download_dir
        print("Download directory: ", self._download_dir)

    def download(self, url: str) -> str:
        """
        Download a url and extract content into a zip file.

        Args:
            url (str): The wiki URL.

        Returns:
            str: The path to the downloaded zip file, or None if the download failed.
        """
        # download wiki urls
        filename = os.path.basename(url)
        output_file = os.path.join(self._download_dir, filename) + ".wiki.gz"

        if os.path.exists(output_file):
            print(f"File '{output_file}' already exists, skipping download.")
            return output_file

        print(f"Downloading txt URLs data from '{url}'...")
        response = requests.get(url)
        html = BeautifulSoup(response.content, "lxml")
        title = html.select("#firstHeading")[0].text
        paragraphs = html.find_all(["p", "ul", "li"])
        intro = "\n".join([para.text for para in paragraphs])
        with gzip.open(output_file, "wt") as file:
            file.write(intro)
        return output_file


class WikitxtIterator(DocumentIterator):
    """
    Wiki document iterator. Will go through the files and parse.
    """

    # The token that separates paragraphs.
    SEPARATOR_TOKEN = "<|endoftext|>"

    def __init__(self):
        super().__init__()
        self._counter = -1

    def iterate(self, file_path):
        self._counter = -1
        file_name = os.path.basename(file_path)

        with gzip.open(file_path, "rt") as file:
            example = []

            def split_meta(example):
                if example:
                    self._counter += 1
                    content = " ".join(example)
                    line_count = content.count("  ") + 1
                    size_in_bytes = len(content.encode("utf-8"))
                    meta = {
                        "id": f"{file_name}-{self._counter}",
                        "file_extension": ".txt",
                        "file_type": "text",
                        "category": "text",
                        "line_count": line_count,
                        "size_in_bytes": size_in_bytes,
                        "path": file_name,
                    }

                    return meta, content

            for line in file:
                if line.strip() == WikitxtIterator.SEPARATOR_TOKEN:
                    if example:
                        yield split_meta(example)
                        example = []
                else:
                    example.append(line.strip())

            if example:
                yield split_meta(example)


class WikitxtExtractor(DocumentExtractor):
    def extract(self, content: str) -> Tuple[Set, str]:
        # No metadata for the text, just the content.
        return {}, content


class GitHubDownloader(DocumentDownloader):
    """
    A class for downloading repositories from GitHub.
    """

    def __init__(self, github_root_dir: str):
        """
        Initializes the DocBuilder object.

        Args:
            github_root_dir: The root directory for GitHub repositories.
        """
        super().__init__()
        # The path under which the repositories will be cloned.
        self.clone_root_dir = os.path.join(github_root_dir, "repos")
        os.makedirs(github_root_dir, exist_ok=True)
        os.makedirs(self.clone_root_dir, exist_ok=True)

    def download(self, url: str) -> str:
        """
        Download a repository as a zip file.

        Args:
            url (str): The URL of the repository.

        Returns:
            str: The path to the downloaded zip file, or None if the download failed.
        """
        repo_name = os.path.basename(url)
        zip_file = os.path.join(self.clone_root_dir, repo_name + ".zip")

        if os.path.exists(zip_file):
            print(f"Repository '{repo_name}' already exists, skipping download.")
            return zip_file

        # Try the common branch names first. A better way to do this would be to
        # query the GitHub API to get the default branch, but that is subject to rate limits.
        success = False

        for branch in ["master", "main"]:
            zip_url = f"https://github.com/{url}/archive/refs/heads/{branch}.zip"

            # Send a GET request to the URL
            response = requests.get(zip_url)

            # Check if the request was successful
            if response.status_code == 200:
                # Write the content of the response to a file
                with open(zip_file, "wb") as file:
                    file.write(response.content)

                # No need to try other branches
                success = True
                break

        if not success:
            print(
                f"Failed to clone repository '{repo_name}' from '{url}' (error code {response.status_code})."
            )
            return None

        return zip_file


class GitHubIterator(DocumentIterator):
    """
    GitHub document iterator. Will go through the files and parse the supported ones.
    """

    # Mapping from file extensions to categories.
    # Will also be used to to ignore irrelevant files.
    SUPPORTED_EXTENSIONS_TO_CATEGORY = {
        ".v": "VerilogVHDL",
        ".vh": "VerilogVHDL",
        ".vhdl": "VerilogVHDL",
        ".va": "VerilogAnalog",
        ".c": "CPP",
        ".cpp": "CPP",
        ".h": "CPP",
        ".hpp": "CPP",
        ".py": "Python",
        ".config": "Config",
        ".mk": "Makefile",
        "makefile": "Makefile",
        "makeppfile": "Makefile",
        ".pm": "Perl",
        ".pl": "Perl",
        ".tcl": "Tcl",
        ".spec": "Spec",
        ".yaml": "Yaml",
        ".yml": "Yaml",
        ".sp": "Spice",
        ".cir": "Spice",
        ".cmd": "Spice",
        ".spf": "Spice",
        ".spice": "Spice",
        ".txt": "text",
        ".json": "text",
        ".xml": "text",
        ".html": "text",
        ".pdf": "text",
        ".md": "text",
        "": "text",  # No extension
    }

    def parse_file(self, zip_ref: ZipFile, file_info: ZipInfo):
        """
        Parses a file from a zip archive and extracts its metadata and content.

        Args:
            zip_ref: The zip archive object.
            file_info: Information about the file in the zip archive.

        Returns:
            A tuple containing the metadata and the content of the file. The metadata is a dictionary.
            If the file extension or filename is not supported, or if the file cannot be decoded,
            None is returned.
        """
        zip_path = zip_ref.filename
        input_fp = file_info.filename
        full_path = os.path.join(zip_path, input_fp)
        # Extract the file name and extension in lower case.
        filename = os.path.basename(input_fp)
        filename_no_ext, ext = os.path.splitext(filename)
        filename_no_ext = filename_no_ext.lower()
        ext = ext.lower()

        # If neither the file extension nor the filename is supported, return None
        if ext not in GitHubIterator.SUPPORTED_EXTENSIONS_TO_CATEGORY:
            if filename_no_ext not in GitHubIterator.SUPPORTED_EXTENSIONS_TO_CATEGORY:
                return None

            # The filename is there, but the extension is not. The category is determined by the filename.
            category = GitHubIterator.SUPPORTED_EXTENSIONS_TO_CATEGORY[filename_no_ext]
        else:
            category = GitHubIterator.SUPPORTED_EXTENSIONS_TO_CATEGORY[ext]

        # Open the file and read its content. Determine the encoding using cchardet. Skip over binary files.
        with zip_ref.open(file_info, "r") as file:
            content = file.read()
            # Determine the encoding of the file
            encoding = chardet.detect(content)["encoding"]

            if not encoding:
                return None

            try:
                content = content.decode(encoding)
            except UnicodeDecodeError:
                # If the file cannot be decoded, return None
                return None

        # Extract the metadata
        line_count = content.count("\n") + 1
        size_in_bytes = file_info.file_size

        if category == "text":
            file_type = "text"
        else:
            file_type = "code"

        metadata = {
            # Use the file path as the unique ID
            "id": full_path,
            "file_extension": ext,
            "file_type": file_type,
            "category": category,
            "line_count": line_count,
            "size_in_bytes": size_in_bytes,
            "path": full_path,
        }
        return metadata, content

    def iterate(self, file_path: str):
        """
        Iterates over the files in a zip archive and yields the parsed content of each file.

        Args:
            file_path: The path to the zip archive.

        Yields:
            Parsed content of each file in the zip archive.
        """

        if not file_path:
            return

        with ZipFile(file_path, "r") as zip_ref:
            for file_info in zip_ref.infolist():
                filename = file_info.filename

                # Skip directories and hidden files
                if file_info.is_dir() or any(
                    part.startswith(".") for part in filename.split(os.sep)
                ):
                    continue

                parsed = self.parse_file(zip_ref, file_info)
                if parsed:
                    yield parsed


class GitHubExtractor(DocumentExtractor):
    def extract(self, content: str):
        # Just return the content.
        return {}, content


class ArxivDownloader(DocumentDownloader):
    """
    A class for downloading article PDFs from arXiv.
    """

    def __init__(self, pdf_root_dir: str):
        """
        Initializes the DocBuilder object.

        Args:
            pdf_root_dir: The root directory for PDF repositories.
        """
        super().__init__()
        # The path under which the pdfs converted to text will be stored.
        self.pdf_root_dir = os.path.join(pdf_root_dir, "pdfs")
        os.makedirs(pdf_root_dir, exist_ok=True)
        os.makedirs(self.pdf_root_dir, exist_ok=True)

    def parse_id(self, input_string: str) -> str:
        """
        Extracts the arXiv ID from a given input string, which can be either an arXiv ID or a URL.

        Args:
            input_string: The input string that is either an arXiv ID or a URL.

        Returns:
            The extracted arXiv ID if the input is valid, otherwise None.
        """
        # Pattern to match a direct arXiv ID
        id_pattern = re.compile(r"\d{4}\.\d{4,5}(v\d+)?$")
        if id_pattern.match(input_string):
            return input_string

        # Pattern to match an arXiv URL and extract the ID
        url_pattern = re.compile(
            r"https?://(?:www\.)?arxiv\.org/(abs|pdf)/(\d{4}\.\d{4,5})(v\d+)?(\.pdf)?$"
        )
        url_match = url_pattern.match(input_string)
        if url_match:
            return url_match.group(2) + (
                url_match.group(3) if url_match.group(3) else ""
            )

        # Raise an error if the input does not match any of the expected formats
        raise ValueError(
            f"The provided input '{input_string}' does not match the expected URL or ID format."
        )

    def download(self, url: str) -> str:
        """
        Downloads the article as a PDF file.

        Args:
            url (str): The URL of the pdf.

        Returns:
            str: The path to the downloaded PDF file, or None if the download failed.
        """
        pdf_name = os.path.basename(url)
        pdf_file = os.path.join(self.pdf_root_dir, pdf_name)

        if os.path.exists(pdf_file):
            print(f"Article '{url}' already exists, skipping download.")
            return pdf_file

        article_id = self.parse_id(url)
        search_result = arxiv.Client().results(arxiv.Search(id_list=[article_id]))

        if article := next(search_result):
            print(f'Downloading arXiv article "{url}"...')
            pdf_path = article.download_pdf(
                dirpath=self.pdf_root_dir, filename=pdf_name
            )
        else:
            print(f"Failed to download article '{url}'.")
            return None

        return pdf_path


class ArxivIterator(DocumentIterator):
    """
    arXiv document iterator. Will go through the files and parse the supported ones.
    """

    def iterate(self, file_path: str):
        """
        Iterates over the pdf files and yields the parsed content of each file.

        Args:
            file_path: The path to the downloaded pdf.

        Yields:
            Parsed content of each file in the pdf.
        """
        if not file_path:
            return

        elements = partition(filename=file_path)

        # Extract the file name and extension in lower case.
        filename = os.path.basename(file_path)
        filename_no_ext, ext = os.path.splitext(filename)
        filename_no_ext = filename_no_ext.lower()
        ext = ext.lower()

        # Read and join the extracted content.
        content = "\n".join([str(el) for el in elements])

        # Extract the metadata
        line_count = content.count("\n") + 1
        size_in_bytes = os.path.getsize(file_path)
        metadata = {
            # Use the file path as the unique ID
            "id": file_path,
            "file_extension": ext,
            "file_type": "text",
            "category": "text",
            "line_count": line_count,
            "size_in_bytes": size_in_bytes,
            "path": file_path,
        }

        yield metadata, content


class ArxivExtractor(DocumentExtractor):
    def extract(self, content: str):
        # Just return the content.
        return {}, content
