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
import random
import string
import tempfile
from datetime import datetime, timedelta

import pandas as pd
import pytest

from nemo_curator.datasets import DocumentDataset


def _generate_dummy_dataset(num_rows: int = 50) -> str:
    # Function to generate a shuffled sequence of integers
    def shuffled_integers(length: int = num_rows) -> int:
        # Create a list of numbers from 0 to length - 1
        integers = list(range(length))

        # Shuffle the list
        random.shuffle(integers)

        # Yield one number from the list each time the generator is invoked
        for integer in integers:
            yield integer

    # Function to generate a random string of a given length
    def generate_random_string(length: int = 10) -> str:
        characters = string.ascii_letters + string.digits  # Alphanumeric characters

        return "".join(random.choice(characters) for _ in range(length))

    # Function to generate a random datetime
    def generate_random_datetime() -> str:
        # Define start and end dates
        start_date = datetime(1970, 1, 1)  # Unix epoch
        end_date = datetime.now()  # Current date

        # Calculate the total number of seconds between the start and end dates
        delta = end_date - start_date
        total_seconds = int(delta.total_seconds())

        # Generate a random number of seconds within this range
        random_seconds = random.randint(0, total_seconds)

        # Add the random number of seconds to the start date to get a random datetime
        random_datetime = start_date + timedelta(seconds=random_seconds)

        # Convert to UTC and format the datetime
        random_datetime_utc = random_datetime.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        return random_datetime_utc

    # Generate the corpus
    corpus = []
    for integer in shuffled_integers():
        corpus.append(
            json.dumps(
                {
                    "id": integer,
                    "date": generate_random_datetime(),
                    "text": generate_random_string(random.randint(5, 100)),
                }
            )
        )

    # Return the corpus
    return "\n".join(corpus)


@pytest.fixture
def jsonl_dataset():
    return _generate_dummy_dataset(num_rows=10)


class TestIO:
    def test_meta_dict(self, jsonl_dataset):
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
            # Write the corpus to the file
            temp_file.write(jsonl_dataset.encode("utf-8"))

            # Flush the data to ensure it's written to disk
            temp_file.flush()

            # Move the cursor to the beginning of the file before reading
            temp_file.seek(0)

            # Read the dataset
            dataset = DocumentDataset.read_json(
                temp_file.name, input_meta={"id": float}
            )

        output_meta = str({col: str(dtype) for col, dtype in dataset.df.dtypes.items()})

        expected_meta = (
            "{'date': 'datetime64[ns, UTC]', 'id': 'float64', 'text': 'object'}"
        )

        assert (
            output_meta == expected_meta
        ), f"Expected: {expected_meta}, got: {output_meta}"

    def test_meta_str(self, jsonl_dataset):
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
            # Write the corpus to the file
            temp_file.write(jsonl_dataset.encode("utf-8"))

            # Flush the data to ensure it's written to disk
            temp_file.flush()

            # Move the cursor to the beginning of the file before reading
            temp_file.seek(0)

            # Read the dataset
            dataset = DocumentDataset.read_json(
                temp_file.name, input_meta='{"id": "float"}'
            )

        output_meta = str({col: str(dtype) for col, dtype in dataset.df.dtypes.items()})

        expected_meta = (
            "{'date': 'datetime64[ns, UTC]', 'id': 'float64', 'text': 'object'}"
        )

        assert (
            output_meta == expected_meta
        ), f"Expected: {expected_meta}, got: {output_meta}"
