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

import argparse

import pandas as pd

from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modules.modify import Modify
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper


def console_script() -> None:
    parser = argparse.ArgumentParser()
    args = ArgumentHelper(parser).add_distributed_args().parse_args()
    client = get_client(**ArgumentHelper.parse_client_args(args))  # noqa: F841

    dataframe = pd.DataFrame({"text": ["Sarah and Ryan went out to play", "Jensen is the CEO of NVIDIA"]})
    dataset = DocumentDataset.from_pandas(dataframe, npartitions=1)

    modifier = PiiModifier(
        log_dir="./logs",
        batch_size=2000,
        language="en",
        supported_entities=["PERSON", "EMAIL_ADDRESS"],
        anonymize_action="replace",
    )

    modify = Modify(modifier)
    modified_dataset = modify(dataset)
    modified_dataset.to_json("output.jsonl")


if __name__ == "__main__":
    console_script()
