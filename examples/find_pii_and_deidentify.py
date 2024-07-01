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

import argparse

import dask.dataframe
import pandas as pd

from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modules.modify import Modify
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper


def console_script():
    parser = argparse.ArgumentParser()
    args = ArgumentHelper(parser).add_distributed_args().parse_args()
    _ = get_client(**ArgumentHelper.parse_client_args(args))

    dataframe = pd.DataFrame(
        {"text": ["Sarah and Ryan went out to play", "Jensen is the CEO of NVIDIA"]}
    )
    dd = dask.dataframe.from_pandas(dataframe, npartitions=1)
    dataset = DocumentDataset(dd)

    modifier = PiiModifier(
        log_dir="./logs",
        batch_size=2000,
        language="en",
        supported_entities=["PERSON", "EMAIL_ADDRESS"],
        anonymize_action="replace",
    )

    modify = Modify(modifier)
    modified_dataset = modify(dataset)
    modified_dataset.df.to_json("output_files/*.jsonl", lines=True, orient="records")


if __name__ == "__main__":
    console_script()
