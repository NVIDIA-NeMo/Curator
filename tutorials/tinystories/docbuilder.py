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
from typing import Set, Tuple

import requests

from nemo_curator.download.doc_builder import (
    DocumentDownloader,
    DocumentExtractor,
    DocumentIterator,
)


class TinyStoriesDownloader(DocumentDownloader):
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

        print(f"Downloading TinyStories dataset from '{url}'...")
        response = requests.get(url)

        with open(output_file, "wb") as file:
            file.write(response.content)

        return output_file


class TinyStoriesIterator(DocumentIterator):
    # The token that separates stories in the TinyStories dataset.
    SEPARATOR_TOKEN = "<|endoftext|>"

    def __init__(self):
        super().__init__()
        self._counter = -1

    def iterate(self, file_path):
        self._counter = -1
        file_name = os.path.basename(file_path)

        with open(file_path, "r") as file:
            example = []

            def split_meta(example):
                if example:
                    self._counter += 1
                    content = " ".join(example)
                    meta = {
                        "file_name": file_name,
                        "id": f"{file_name}-{self._counter}",
                    }

                    return meta, content

            for line in file:
                if line.strip() == TinyStoriesIterator.SEPARATOR_TOKEN:
                    if example:
                        yield split_meta(example)
                        example = []
                else:
                    example.append(line.strip())

            if example:
                yield split_meta(example)


class TinyStoriesExtractor(DocumentExtractor):
    def extract(self, content: str) -> Tuple[Set, str]:
        # No metadata for the text, just the content.
        return {}, content
