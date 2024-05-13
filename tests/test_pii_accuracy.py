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

import logging
import re
from pathlib import Path

import pandas as pd
from dask import dataframe as dd
from dask.distributed import Client, LocalCluster

import nemo_curator as nc
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import DocumentFilter
from nemo_curator.modifiers import PiiModifier
from nemo_curator.pii.algorithm import PiiDeidentifier
from nemo_curator.utils.decorators import batched

LOGGER = logging.getLogger(__name__)


def load_test_cases(filename):
    filepath = Path("tests/pii_data") / filename
    with open(filepath) as fp:
        data = fp.readlines()

    raw_data = [
        (re.sub(r"<[^>]*>([^<]*)</[^>]*>", r"\1", line)).strip() for line in data
    ]
    masked_data = [
        (
            re.sub(r"(<[^>]*>([^<]*)</[^>]*>)", lambda x: "*" * len(x.group(2)), line)
        ).strip()
        for line in data
    ]

    return list(zip(raw_data, masked_data))


def compare_outputs(output1, output2):
    output1 = re.sub(r"\*[\* ]*", "*****", output1)
    output2 = re.sub(r"\*[\* ]*", "*****", output2)
    return output1 == output2


def generate_single_category_test(category, filename):
    deidentifier = PiiDeidentifier("en", [category], "mask")
    test_data = load_test_cases(filename)

    for input, target in test_data:
        output = deidentifier.deidentify_text(input)
        print("============================")
        print("Input: ", input)
        print("Output: ", output)
        print("Expected Output: ", target)
        print("Matches:", "No" if output != target else "Yes")
        assert output == target


class TestPIIAccuracy:
    def test_email(self):
        generate_single_category_test("EMAIL_ADDRESS", "emails.txt")

    def test_ip_address(self):
        generate_single_category_test("IP_ADDRESS", "ip_address.txt")

    def test_address(self):
        generate_single_category_test("ADDRESS", "address.txt")

    def test_ssn(self):
        generate_single_category_test("US_SSN", "ssn.txt")

    def test_birthdates(self):
        generate_single_category_test("DATE_TIME", "birthdates.txt")

    def test_card_no(self):
        generate_single_category_test("CREDIT_CARD", "card_no.txt")

    def test_names(self):
        generate_single_category_test("PERSON", "names.txt")

    def test_phone_numbers(self):
        generate_single_category_test("PHONE_NUMBER", "phone_numbers.txt")

    def test_multiple(self):
        deidentifier = PiiDeidentifier("en", anonymize_action="mask")
        test_data = load_test_cases("multiple.txt")

        for input, target in test_data:
            output = deidentifier.deidentify_text(input)
            print("============================")
            print("Input: ", input)
            print("Output: ", output)
            print("Expected Output: ", target)
            match = compare_outputs(output, target)
            output1 = re.sub(r"\*[\* ]*", "*****", output)
            output2 = re.sub(r"\*[\* ]*", "*****", target)
            print(output1)
            print(output2)
            print("match value: ", match)
            print("Matches:", "No" if not match else "Yes")
            assert match == True

    def test_batch_accuracy(self):
        deidentifier = PiiDeidentifier("en", anonymize_action="mask")
        test_data = load_test_cases("multiple.txt")
        inputs = [data[0] for data in test_data]
        targets = [data[1] for data in test_data]
        outputs = deidentifier.deidentify_text_batch(inputs)
        print("Inputs: ", inputs)
        print("Output: ", outputs)
        print("Expected Outputs: ", targets)

        match = all(compare_outputs(x, y) for x, y in zip(outputs, targets))
        print("Matches:", "No" if not match else "Yes")
        assert match == True


class BatchedLengthFilter(DocumentFilter):
    """
    Keeps documents of a given length
    """

    def __init__(self, min_length=5, max_length=10):
        super().__init__()
        self.min_length = min_length
        self.max_length = max_length

    @batched
    def score_document(self, df):
        return df.str.len()

    @batched
    def keep_document(self, scores):
        min_threshold = self.min_length <= scores
        max_threshold = scores <= self.max_length
        return min_threshold & max_threshold


class TestPIIModule:
    def test_filter_chain(self):
        inputs = [
            "Alice goes on a walk",
            "Bob goes on a walk",
            "Someone named Charlie goes on a walk",
            "A human walking is David",
            "A human walking is Eliza",
        ]
        targets = [
            "***** goes on a walk",
            "*** goes on a walk",
            "A human walking is *****",
            "A human walking is *****",
        ]
        input_df = pd.DataFrame({"text": inputs})
        target_df = pd.DataFrame({"text": targets})
        with LocalCluster(n_workers=1, threads_per_worker=1) as cluster:
            with Client(cluster):
                input_dataset = DocumentDataset(dd.from_pandas(input_df, npartitions=1))
                pipeline = nc.Sequential(
                    [
                        nc.ScoreFilter(
                            BatchedLengthFilter(min_length=0, max_length=25)
                        ),
                        nc.Modify(
                            PiiModifier(
                                language="en", anonymize_action="mask", device="cpu"
                            )
                        ),
                    ]
                )
                output_dataset = pipeline(input_dataset)

                output_df = output_dataset.df.compute().reset_index(drop=True)
                match = all(output_df["text"] == target_df["text"])
                assert match
