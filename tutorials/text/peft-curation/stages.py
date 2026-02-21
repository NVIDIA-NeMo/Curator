# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from collections.abc import Iterator
from typing import Any

import requests
from loguru import logger
from transformers import AutoTokenizer

from nemo_curator.stages.text.download.base import (
    DocumentDownloader,
    DocumentDownloadExtractStage,
    DocumentExtractor,
    DocumentIterator,
    URLGenerator,
)
from nemo_curator.stages.text.filters import DocumentFilter
from nemo_curator.stages.text.modifiers import DocumentModifier


class EnronEmailsURLGenerator(URLGenerator):
    """
    The implementation of a URL generator. For this example, we only need to download a single file.
    """

    def generate_urls(self) -> list[str]:
        return [
            "https://huggingface.co/datasets/neelblabla/enron_labeled_emails_with_subjects-llama2-7b_finetuning/raw/main/prompts_train.csv"
        ]


class EnronEmailsDownloader(DocumentDownloader):
    """
    The implementation of a document downloader. Fetches the given URL and populates the given file.
    """

    def _get_output_filename(self, url: str) -> str:
        """Generate output filename from URL.

        Args:
            url: URL to download

        Returns:
            Output filename (without directory path)
        """
        return os.path.basename(url)

    def _download_to_path(self, url: str, path: str) -> tuple[bool, str | None]:
        """Download URL to specified path.

        Args:
            url: URL to download
            path: Local path to save file

        Returns:
            Tuple of (success, error_message). If success is True, error_message should be None.
            If success is False, error_message should contain the error details.
        """
        logger.debug(f"Downloading Enron Emails dataset from '{url}'...")
        response = requests.get(url)  # noqa: S113

        with open(path, "wb") as file:
            file.write(response.content)

        return True, None


class EnronEmailsIterator(DocumentIterator):
    """
    The implementation of an iterator defining how to itereate the raw dataset and fetch records.
    """

    def __init__(self):
        super().__init__()
        # The regular expression pattern to extract each email.
        self._pattern_email = re.compile(r"\"<s>.*?<s>\"", re.DOTALL)
        self._counter = -1

    def iterate(self, file_path: str) -> Iterator[tuple[dict[str, str], str]]:
        self._counter = -1

        with open(file_path, encoding="utf-8") as file:
            lines = file.readlines()

        # Ignore the first line which contains the header.
        file_content = "".join(lines[1:])
        # Find all the emails in the file.
        it = self._pattern_email.finditer(file_content)

        for email in it:
            self._counter += 1
            content = email.group().strip('"').strip()
            yield {
                "id": f"email-{self._counter}",
                "raw_content": content,
            }

    def output_columns(self) -> list[str]:
        return ["id", "raw_content"]


class EnronEmailsExtractor(DocumentExtractor):
    """
    The implementation of a document extractor. For this example, we use a regex to find each part.
    """

    def __init__(self):
        # The regular expression pattern to extract subject/body/label into groups.
        self._pattern_email_parts = re.compile(
            r"Subject:: (.*?)\nBody:: (.*?)\n.*\[/INST\] (.*?) <s>",
            re.DOTALL,
        )

    def extract(self, record: dict[str, str]) -> dict[str, Any] | None:
        content = record["raw_content"]
        matches = self._pattern_email_parts.findall(content)

        if not matches:
            return None

        matches = matches[0]

        return {
            "id": record["id"],
            "subject": matches[0].strip(),
            "body": matches[1].strip(),
            "category": matches[2].strip(),
        }

    def input_columns(self) -> list[str]:
        return ["id", "raw_content"]

    def output_columns(self) -> list[str]:
        return ["id", "subject", "body", "category"]


class EnronEmailsDownloadExtractStage(DocumentDownloadExtractStage):
    """
    The implementation of the download and extraction stage. Combines the above into a single stage.
    """

    def __init__(
        self,
        download_dir: str,
        verbose: bool = True,
    ):
        self.name = "enron_emails_download_extract_pipeline"

        self.url_generator = EnronEmailsURLGenerator()
        self.downloader = EnronEmailsDownloader(
            download_dir=download_dir,
            verbose=verbose,
        )
        self.iterator = EnronEmailsIterator()
        self.extractor = EnronEmailsExtractor()

        super().__init__(
            url_generator=self.url_generator,
            downloader=self.downloader,
            iterator=self.iterator,
            extractor=self.extractor,
            add_filename_column=True,
        )

    def get_description(self) -> str:
        """Get a description of this composite stage."""
        return "Enron Emails pipeline"


class FilterEmailsWithLongBody(DocumentFilter):
    """
    If the email is too long, discard.
    """

    def __init__(self, max_length: int = 5000):
        super().__init__()
        self.max_length = max_length

    def score_document(self, text: str) -> bool:
        return len(text) <= self.max_length

    def keep_document(self, score: bool) -> bool:
        return score


class FilterEmptyEmails(DocumentFilter):
    """
    Detects empty emails (either empty body, or labeled as empty). Returns `True` for empty emails.
    """

    def score_document(self, text: str) -> bool:
        return (
            not isinstance(text, str)  # The text is not a string
            or len(text.strip()) == 0  # The text is empty
            or "Empty message" in text  # The email is labeled as empty
        )

    def keep_document(self, score: bool) -> bool:
        return score


class AddPeriod(DocumentModifier):
    """
    A simple modifier that adds a period to the end of each email category.
    """

    def modify_document(self, text: str) -> str:
        return text + "."


class ApplyChatTemplate(DocumentModifier):
    """
    A modifier that takes a tokenizer and formats training examples using a chat template.
    """

    def __init__(self, tokenizer: AutoTokenizer) -> None:
        super().__init__()
        self._tokenizer = tokenizer

    def modify_document(self, subject: str, body: str, category: str) -> str:
        messages = [
            {
                "role": "system",
                "content": """You are reviewing emails. Your task is to look at the subject and the body of an email, then output a category that best describes the email.
Always respond in this format, and ensure to place a period at the end of the category:

# Category
<your output category>.

""",
            },
            {
                "role": "user",
                "content": f"""# Subject
{subject}

# Body
{body}

""",
            },
            {
                "role": "assistant",
                "content": f"""# Category
{category}

""",
            },
        ]
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )


class AddTokenCount(DocumentModifier):
    """
    A basic modifier that counts the number of tokens in the "text" column.
    """

    def __init__(self, tokenizer: AutoTokenizer) -> None:
        super().__init__()
        self._tokenizer = tokenizer

    def modify_document(self, text: str) -> str:
        return len(self._tokenizer(text).input_ids)
