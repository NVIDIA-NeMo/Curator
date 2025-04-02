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

import pytest

from nemo_curator.modifiers.async_llm_pii_modifier import AsyncLLMPiiModifier
from nemo_curator.modifiers.llm_pii_modifier import LLMPiiModifier
from nemo_curator.utils.llm_pii_utils import (
    EntitySpan,
    find_entity_spans,
    fix_overlaps,
    get_system_prompt,
    redact,
    validate_entity,
    validate_keys,
)


class TestLLMPiiUtils:
    def test_validate_keys(self):
        valid_keys = {"entity_type": "date_time", "entity_text": "1989.12.22"}
        assert validate_keys(valid_keys)

        incomplete_keys = {"entity_type": "date_time"}
        assert not validate_keys(incomplete_keys)

        invalid_keys = {"type": "date_time", "text": "1989.12.22"}
        assert not validate_keys(invalid_keys)

    def test_validate_entity(self):
        valid_entity = {"entity_type": "date_time", "entity_text": "1989.12.22"}
        text = "Unloading Plan for Shipment MRN-293104, MED25315002, dated 1989.12.22. Driver EMP730359, Vehicle KS40540825."
        min_length = 2
        assert validate_entity(valid_entity, text, min_length)

        invalid_entity = {"type": "date_time", "text": "1989.12.22"}
        assert not validate_entity(invalid_entity, text, min_length)

        invalid_entity_text = {"entity_type": "date_time", "entity_text": "1999.12.22"}
        assert not validate_entity(invalid_entity_text, text, min_length)

        short_entity_text = {"entity_type": "date_time", "entity_text": "1"}
        assert not validate_entity(short_entity_text, text, min_length)

    def test_redact(self):
        full_text = "Transaction details: gasLimit set to 1000000 units by tw_brian740, gasPrice set to 10 Gwei by veronicawood@example.org, contactable at +1-869-341-9301x7005, located at Suite 378, Yolanda Mountain, Burkeberg."
        pii_entities = [
            {"entity_type": "name", "entity_text": "tw_brian740"},
            {"entity_type": "name", "entity_text": "veronicawood"},
            {"entity_type": "email", "entity_text": "veronicawood@example.org"},
            {"entity_type": "phone_number", "entity_text": "+1-869-341-9301x7005"},
            {
                "entity_type": "location",
                "entity_text": "Suite 378, Yolanda Mountain, Burkeberg",
            },
        ]
        redacted_text = redact(full_text, pii_entities)
        expected_text = "Transaction details: gasLimit set to 1000000 units by {{name}}, gasPrice set to 10 Gwei by {{email}}, contactable at {{phone_number}}, located at {{location}}."
        assert redacted_text == expected_text

    def test_find_entity_spans(self):
        text = "Transaction details: gasLimit set to 1000000 units by tw_brian740, gasPrice set to 10 Gwei by veronicawood@example.org, contactable at +1-869-341-9301x7005, located at Suite 378, Yolanda Mountain, Burkeberg."
        entities = [
            {"entity_type": "name", "entity_text": "tw_brian740"},
            # Add repeated entity to check robustness
            {"entity_type": "name", "entity_text": "tw_brian740"},
            {"entity_type": "name", "entity_text": "veronicawood"},
            {"entity_type": "email", "entity_text": "veronicawood@example.org"},
            {"entity_type": "phone_number", "entity_text": "+1-869-341-9301x7005"},
            {
                "entity_type": "location",
                "entity_text": "Suite 378, Yolanda Mountain, Burkeberg",
            },
        ]
        spans = find_entity_spans(text, entities)
        assert spans == [
            EntitySpan(entity_type="name", start_position=54, end_position=65),
            EntitySpan(entity_type="name", start_position=94, end_position=106),
            EntitySpan(entity_type="email", start_position=94, end_position=118),
            EntitySpan(
                entity_type="phone_number", start_position=135, end_position=155
            ),
            EntitySpan(entity_type="location", start_position=168, end_position=206),
        ]

    def test_fix_overlaps(self):
        assert fix_overlaps([]) == []

        spans = [
            EntitySpan(entity_type="name", start_position=54, end_position=65),
            EntitySpan(entity_type="name", start_position=94, end_position=106),
            EntitySpan(entity_type="email", start_position=94, end_position=118),
            EntitySpan(
                entity_type="phone_number", start_position=135, end_position=155
            ),
            EntitySpan(entity_type="location", start_position=168, end_position=206),
        ]
        fixed_spans = fix_overlaps(spans)
        assert fixed_spans == [
            EntitySpan(entity_type="name", start_position=54, end_position=65),
            EntitySpan(entity_type="email", start_position=94, end_position=118),
            EntitySpan(
                entity_type="phone_number", start_position=135, end_position=155
            ),
            EntitySpan(entity_type="location", start_position=168, end_position=206),
        ]

        spans = [
            EntitySpan(entity_type="date_time", start_position=59, end_position=69),
            EntitySpan(entity_type="employee_id", start_position=78, end_position=87),
            EntitySpan(
                entity_type="medical_record_number", start_position=28, end_position=38
            ),
            EntitySpan(
                entity_type="vehicle_identifier", start_position=97, end_position=107
            ),
        ]
        fixed_spans = fix_overlaps(spans)
        assert fixed_spans == [
            EntitySpan(
                entity_type="medical_record_number", start_position=28, end_position=38
            ),
            EntitySpan(entity_type="date_time", start_position=59, end_position=69),
            EntitySpan(entity_type="employee_id", start_position=78, end_position=87),
            EntitySpan(
                entity_type="vehicle_identifier", start_position=97, end_position=107
            ),
        ]


class TestLLMPiiModifier:
    def test_system_prompt(self):
        # Default system prompt, default PII labels, and English
        modifier = LLMPiiModifier(base_url="https://integrate.api.nvidia.com/v1")
        assert modifier.system_prompt == get_system_prompt()

        # Default system prompt, default PII labels, and non-English
        with pytest.warns(
            UserWarning,
            match="The default system prompt is only available for English text",
        ):
            modifier = LLMPiiModifier(
                base_url="https://integrate.api.nvidia.com/v1", language="fr"
            )
            assert modifier.system_prompt == get_system_prompt()

        # Custom system prompt, default PII labels, and English
        with pytest.warns(
            UserWarning,
            match="Using the default system prompt is strongly recommended for English text",
        ):
            system_prompt = "You are a helpful assistant."
            modifier = LLMPiiModifier(
                base_url="https://integrate.api.nvidia.com/v1",
                system_prompt=system_prompt,
            )
            assert modifier.system_prompt == system_prompt

        # Custom system prompt, default PII labels, and non-English
        system_prompt = "Vous êtes un assistant utile."
        modifier = LLMPiiModifier(
            base_url="https://integrate.api.nvidia.com/v1", system_prompt=system_prompt
        )
        assert modifier.system_prompt == system_prompt

        # Default system prompt and custom PII labels
        pii_labels = ["name", "email", "phone_number", "location"]
        modifier = LLMPiiModifier(
            base_url="https://integrate.api.nvidia.com/v1", pii_labels=pii_labels
        )
        assert modifier.system_prompt == get_system_prompt(pii_labels)

        # Custom system prompt and custom PII labels
        with pytest.warns(
            UserWarning,
            match="Custom system_prompt and custom pii_labels were both provided",
        ):
            system_prompt = "You are a helpful assistant."
            pii_labels = ["name", "email", "phone_number", "location"]
            modifier = LLMPiiModifier(
                base_url="https://integrate.api.nvidia.com/v1",
                system_prompt=system_prompt,
                pii_labels=pii_labels,
            )
            assert modifier.system_prompt == system_prompt


class TestAsyncLLMPiiModifier:
    def test_system_prompt(self):
        # Default system prompt, default PII labels, and English
        modifier = AsyncLLMPiiModifier(base_url="https://integrate.api.nvidia.com/v1")
        assert modifier.system_prompt == get_system_prompt()

        # Default system prompt, default PII labels, and non-English
        with pytest.warns(
            UserWarning,
            match="The default system prompt is only available for English text",
        ):
            modifier = AsyncLLMPiiModifier(
                base_url="https://integrate.api.nvidia.com/v1", language="fr"
            )
            assert modifier.system_prompt == get_system_prompt()

        # Custom system prompt, default PII labels, and English
        with pytest.warns(
            UserWarning,
            match="Using the default system prompt is strongly recommended for English text",
        ):
            system_prompt = "You are a helpful assistant."
            modifier = AsyncLLMPiiModifier(
                base_url="https://integrate.api.nvidia.com/v1",
                system_prompt=system_prompt,
            )
            assert modifier.system_prompt == system_prompt

        # Custom system prompt, default PII labels, and non-English
        system_prompt = "Vous êtes un assistant utile."
        modifier = AsyncLLMPiiModifier(
            base_url="https://integrate.api.nvidia.com/v1", system_prompt=system_prompt
        )
        assert modifier.system_prompt == system_prompt

        # Default system prompt and custom PII labels
        pii_labels = ["name", "email", "phone_number", "location"]
        modifier = AsyncLLMPiiModifier(
            base_url="https://integrate.api.nvidia.com/v1", pii_labels=pii_labels
        )
        assert modifier.system_prompt == get_system_prompt(pii_labels)

        # Custom system prompt and custom PII labels
        with pytest.warns(
            UserWarning,
            match="Custom system_prompt and custom pii_labels were both provided",
        ):
            system_prompt = "You are a helpful assistant."
            pii_labels = ["name", "email", "phone_number", "location"]
            modifier = AsyncLLMPiiModifier(
                base_url="https://integrate.api.nvidia.com/v1",
                system_prompt=system_prompt,
                pii_labels=pii_labels,
            )
            assert modifier.system_prompt == system_prompt
