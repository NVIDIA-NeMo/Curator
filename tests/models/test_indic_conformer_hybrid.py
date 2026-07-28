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

from pathlib import Path
from types import SimpleNamespace

from nemo_curator.models.indic_conformer_hybrid import IndicConformerHybridASR


def test_local_nemo_path_is_used_without_hub_download(tmp_path: Path) -> None:
    checkpoint = tmp_path / "indic.nemo"
    checkpoint.touch()

    assert IndicConformerHybridASR._resolve_nemo_path(str(checkpoint)) == str(checkpoint)


def test_local_token_ids_are_mapped_through_aggregate_tokenizer() -> None:
    adapter = IndicConformerHybridASR("checkpoint.nemo")
    tokenizer = SimpleNamespace(
        token_id_offset={"hi": 100},
        ids_to_text=lambda ids: f"tokens={ids}",
    )
    adapter._model = SimpleNamespace(tokenizer=tokenizer)

    assert adapter._ids_to_text([1, 2], "hi") == "tokens=[101, 102]"


def test_empty_token_sequence_decodes_to_empty_text() -> None:
    adapter = IndicConformerHybridASR("checkpoint.nemo")

    assert adapter._ids_to_text([], "hi") == ""
