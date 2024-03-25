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

import dask.dataframe as dd
import pandas as pd

import nemo_curator
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers import UnicodeReformatter


def list_to_dataset(documents, col_name="text", npartitions=2):
    data = {col_name: documents}
    pdf = pd.DataFrame(data)

    return DocumentDataset(dd.from_pandas(pdf, npartitions=npartitions))


class TestUnicodeReformatter:
    def test_reformatting(self):
        # Examples taken from ftfy documentation:
        # https://ftfy.readthedocs.io/en/latest/
        dataset = list_to_dataset(
            [
                "âœ” No problems",
                "The Mona Lisa doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t have eyebrows.",
                "l’humanitÃ©",
                "Ã perturber la rÃ©flexion",
                "Clean document already.",
            ]
        )
        expected_results = [
            "✔ No problems",
            "The Mona Lisa doesn't have eyebrows.",
            "l'humanité",
            "à perturber la réflexion",
            "Clean document already.",
        ]
        expected_results.sort()

        modifier = nemo_curator.Modify(UnicodeReformatter())
        fixed_dataset = modifier(dataset)
        actual_results = fixed_dataset.df.compute()["text"].to_list()
        actual_results.sort()

        assert (
            expected_results == actual_results
        ), f"Expected: {expected_results}, but got: {actual_results}"
