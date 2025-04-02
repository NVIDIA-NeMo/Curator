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

import dask.dataframe
import pandas as pd

from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers.llm_pii_modifier import LLMPiiModifier
from nemo_curator.modules.modify import Modify
from nemo_curator.utils.distributed_utils import get_client


def console_script():
    _ = get_client()

    dataframe = pd.DataFrame(
        {
            "text": [
                # Sampled from https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1
                "Transaction details: gasLimit set to 1000000 units by tw_brian740, gasPrice set to 10 Gwei by veronicawood@example.org, contactable at +1-869-341-9301x7005, located at Suite 378, Yolanda Mountain, Burkeberg.",
                "Unloading Plan for Shipment MRN-293104, MED25315002, dated 1989.12.22. Driver EMP730359, Vehicle KS40540825.",
            ]
        }
    )
    dataset = DocumentDataset.from_pandas(dataframe, npartitions=1)

    modifier = LLMPiiModifier(
        # Endpoint for the user's NIM
        base_url="http://0.0.0.0:8000/v1",
        api_key="API KEY (if needed)",
        model="meta/llama-3.1-70b-instruct",
    )

    modify = Modify(modifier)
    modified_dataset = modify(dataset)
    modified_dataset.to_json("output.jsonl")


if __name__ == "__main__":
    console_script()
