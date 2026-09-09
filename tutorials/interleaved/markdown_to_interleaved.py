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

"""Convert one PIN-14M markdown document to interleaved rows."""

import pandas as pd
from datasets import load_dataset

from nemo_curator.stages.interleaved import MarkdownToInterleavedStage
from nemo_curator.tasks import DocumentBatch


def main() -> None:
    dataset = load_dataset("m-a-p/PIN-14M", "pin", split="train", streaming=True)
    sample = next(iter(dataset.skip(94)))
    documents = DocumentBatch(dataset_name="PIN-14M", data=pd.DataFrame([sample]))

    interleaved = MarkdownToInterleavedStage().process(documents)
    print(interleaved.to_pandas()[["sample_id", "position", "modality", "text_content", "source_ref"]])


if __name__ == "__main__":
    main()
