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

import json
import os
import re
import warnings
from collections.abc import Iterator

import requests
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

from nemo_curator.download.doc_builder import (
    DocumentDownloader,
    DocumentExtractor,
    DocumentIterator,
)

# Ignore the specific BeautifulSoup warning
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


class LawQADownloader(DocumentDownloader):
    """
    A class for downloading Law QA dataset.
    """

    def __init__(self, download_dir: str):
        super().__init__()

        if not os.path.isdir(download_dir):
            os.makedirs(download_dir)

        self._download_dir = download_dir
        print("Download directory: ", self._download_dir)

    def download(self, url: str) -> str:
        """
        Downloads the Law QA dataset from the given URL.

        Args:
            url (str): The URL of the Law QA dataset.

        Returns:
            str: The path of the downloaded file.

        """
        filename = os.path.basename(url)
        output_file = os.path.join(self._download_dir, filename)

        if os.path.exists(output_file):
            print(f"File '{output_file}' already exists, skipping download.")
            return output_file

        print(f"Downloading Law QA dataset from '{url}'...")
        response = requests.get(url)  # noqa: S113

        with open(output_file, "wb") as file:
            file.write(response.content)

        return output_file


class LawQAIterator(DocumentIterator):
    def __init__(self):
        super().__init__()
        self._counter = -1
        self._extractor = LawQAExtractor()

    def iterate(self, file_path: str) -> Iterator[dict[str, str]]:
        """
        Iterates over the content of a file and yields extracted records.

        Args:
            file_path (str): The path to the file to be iterated.

        Yields:
            dict: A dictionary representing a record extracted from the file.
        """
        self._counter = -1
        file_name = os.path.basename(file_path)

        with open(file_path, encoding="utf-8") as file:
            lines = file.readlines()

        file_content = "".join(lines)
        json_content = json.loads(file_content)

        for row in json_content:
            self._counter += 1
            extracted_content = self._extractor.extract(row)

            # Skip if the question has no answers.
            if extracted_content is None:
                continue

            _id, extracted_content = extracted_content
            meta = {
                "file_name": file_name,
                "id": f"law-stackexchange-qa-{_id}",
            }

            record = {**meta, **extracted_content}
            yield record


class LawQAExtractor(DocumentExtractor):
    def extract(self, content: dict[str, str]) -> dict[str, str]:
        """
        Extracts relevant information from a law-related question and its best answer.

        Args:
            content (str): The content of the question and its answers.

        Returns:
            Dict[str, str]: A dictionary containing the extracted information, including the question ID, title, body,
            score, best answer, best answer score, and tags.
        """
        _id = content["question_id"]
        q_title = content["question_title"]
        q_body = content["question_body"]
        q_score = content["score"]
        tags = ",".join(sorted(content["tags"]))

        # If this question has no answers, skip it.
        if len(content["answers"]) == 0:
            return None

        # All answers are sorted by votes, so take the first answer as the best one.
        best_answer = content["answers"][0]
        best_answer_score = best_answer["score"]
        best_answer = best_answer["body"]

        # Get rid of HTML tags using beautifulsoup
        # NOTE: Doing this here so that I can split the dataset without having to worry about curating the test split.
        q_title = self._clean_html(q_title)
        q_body = self._clean_html(q_body)
        best_answer = self._clean_html(best_answer)

        return _id, {
            "title": q_title,
            "question": q_body,
            "question_score": q_score,
            "answer": best_answer,
            "answer_score": best_answer_score,
            "tags": tags,
        }

    def _clean_html(self, text: str) -> str:
        text = BeautifulSoup(text, "lxml").get_text()
        return re.sub(r"\s+", " ", text).strip()
