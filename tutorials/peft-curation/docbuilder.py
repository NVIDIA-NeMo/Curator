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
import re
from typing import Dict

import requests

from nemo_curator.download.doc_builder import (
    DocumentDownloader,
    DocumentExtractor,
    DocumentIterator,
)


class EmailsDownloader(DocumentDownloader):
    def __init__(self, download_dir: str):
        super().__init__()

        if not os.path.isdir(download_dir):
            os.makedirs(download_dir)

        self._download_dir = download_dir
        print("Download directory: ", self._download_dir)

    def download(self, url: str) -> str:
        filename = os.path.basename(url)
        output_file = os.path.join(self._download_dir, filename)

        if os.path.exists(output_file):
            print(f"File '{output_file}' already exists, skipping download.")
            return output_file

        print(f"Downloading Enron emails dataset from '{url}'...")
        response = requests.get(url)

        with open(output_file, "wb") as file:
            file.write(response.content)

        return output_file


class EmailsIterator(DocumentIterator):

    def __init__(self):
        super().__init__()
        self._counter = -1
        self._extractor = EmailsExtractor()
        # The regular expression pattern to extract each email.
        self._pattern = re.compile(r"\"<s>.*?<s>\"", re.DOTALL)

    def iterate(self, file_path):
        self._counter = -1
        file_name = os.path.basename(file_path)

        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # Ignore the first line which contains the header.
        file_content = "".join(lines[1:])
        # Find all the emails in the file.
        it = self._pattern.finditer(file_content)

        for email in it:
            self._counter += 1
            content = email.group().strip('"').strip()
            meta = {
                "file_name": file_name,
                "id": f"email-{self._counter}",
            }
            extracted_content = self._extractor.extract(content)

            # Skip if no content extracted
            if not extracted_content:
                continue

            record = {**meta, **extracted_content}
            yield record


class EmailsExtractor(DocumentExtractor):
    def __init__(self):
        super().__init__()
        # The regular expression pattern to extract subject/body/label into groups.
        self._pattern = re.compile(
            r"Subject:: (.*?)\nBody:: (.*?)\n.*\[/INST\] (.*?) <s>", re.DOTALL
        )

    def extract(self, content: str) -> Dict[str, str]:
        matches = self._pattern.findall(content)

        if not matches:
            return None

        matches = matches[0]

        return {
            "subject": matches[0].strip(),
            "body": matches[1].strip(),
            "category": matches[2].strip(),
        }
