# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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


import pytest

from nemo_curator.stages.audio.agent import build_contract, describe_stage, to_json_schema
from nemo_curator.stages.audio.common import PreserveByValueConditionsStage

jsonschema = pytest.importorskip("jsonschema")


@pytest.mark.parametrize(
    "conditions",
    [
        {"duration": {"operator": "gt", "target_value": 1}},
        [{"input_value_key": "duration", "operator": "gt", "target_value": 1}],
    ],
)
def test_condition_schema_accepts_both_runtime_forms(conditions: dict[str, object] | list[dict[str, object]]) -> None:
    schema = to_json_schema(describe_stage("PreserveByValueConditionsStage").params)
    jsonschema.validate({"conditions": conditions}, schema)
    assert build_contract(PreserveByValueConditionsStage(conditions)).cardinality == "filter"


def test_optional_parameter_schema_accepts_explicit_null() -> None:
    schema = to_json_schema(describe_stage("SampleRateFilterStage").params)
    jsonschema.validate({"allowed_sample_rates": None, "min_sample_rate": None}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"allowed_sample_rates": "16000"}, schema)
