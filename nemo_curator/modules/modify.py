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

from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers import DocumentModifier
from nemo_curator.utils.module_utils import is_batched


class Modify:
    def __init__(self, modifier: DocumentModifier, text_field="text"):
        self.modifier = modifier
        self.text_field = text_field

    def __call__(self, dataset: DocumentDataset) -> DocumentDataset:
        if is_batched(self.modifier.modify_document):
            dataset.df[self.text_field] = dataset.df[self.text_field].map_partitions(
                self.modifier.modify_document, meta=(None, str)
            )
        else:
            dataset.df[self.text_field] = dataset.df[self.text_field].apply(
                self.modifier.modify_document, meta=(None, str)
            )

        return dataset
