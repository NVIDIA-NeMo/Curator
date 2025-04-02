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

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Tuple


@dataclass
class EntitySpan:
    entity_type: str
    start_position: int
    end_position: int


JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["entity_type", "entity_text"],
        "properties": {
            "entity_type": {"type": "string"},
            "entity_text": {"type": "string"},
        },
    },
}

# Recommended PII entities to redact
PII_LABELS = [
    "medical_record_number",
    "location",
    "address",
    "ssn",
    "date_of_birth",
    "date_time",
    "name",
    "email",
    "customer_id",
    "employee_id",
    "phone_number",
    "ip_address",
    "credit_card_number",
    "user_name",
    "device_identifier",
    "bank_routing_number",
    "company_name",
    "unique_identifier",
    "biometric_identifier",
    "account_number",
    "certificate_license_number",
    "license_plate",
    "vehicle_identifier",
    "api_key",
    "password",
    "health_plan_beneficiary_number",
    "national_id",
    "tax_id",
    "url",
    "swift_bic",
    "cvv",
    "pin",
]


def get_system_prompt(pii_labels: List[str] = PII_LABELS) -> str:
    return (
        "You are an expert redactor. The user is going to provide you with "
        "some text. Please find all personally identifying information from "
        "this text. Return results according to this JSON schema: "
        f"{str(JSON_SCHEMA)}\nOnly return results for entities which actually "
        "appear in the text. It is very important that you return the  "
        "entity_text by copying it exactly from the input. Do not perform any "
        "modification or normalization of the text. The entity_type should be "
        f"one of these: {', '.join(pii_labels)}\n"
    )


def validate_entity(
    entity: Dict[str, str],
    text: str,
    min_length: int = 2,
) -> bool:
    """Validate entity"""

    if not validate_keys(entity):
        return False

    if entity["entity_text"] not in text:
        return False

    if len(entity["entity_text"]) < min_length:
        return False

    return True


def validate_keys(entity_dict: Dict) -> bool:
    """Validate that keys in entity dict match schema"""

    required_keys = JSON_SCHEMA["items"]["required"]

    entity_keys = list(entity_dict.keys())
    if len(entity_keys) != len(required_keys):
        return False

    for k in required_keys:
        if k not in entity_keys:
            return False

    return True


def redact(
    full_text: str,
    pii_entities: List[Dict[str, str]],
) -> str:
    """Redact given entities from the original text"""

    entity_spans = find_entity_spans(full_text, pii_entities)

    # Initialize an offset to track the changes in the text
    redacted_text = full_text
    offset = 0
    for entity in fix_overlaps(entity_spans):
        entity_type = entity.entity_type
        start_position = entity.start_position + offset
        end_position = entity.end_position + offset

        # Replace the entity value with its type
        replacement = "{{" + entity_type + "}}"
        redacted_text = (
            redacted_text[:start_position] + replacement + redacted_text[end_position:]
        )

        # Update the offset
        offset += len(replacement) - (end_position - start_position)

    return redacted_text


def find_entity_spans(text: str, entities: List[Dict[str, str]]) -> List[EntitySpan]:
    """
    Find the start and end indexes for each entity in the given text.

    Args:
        text (str): The input text string.
        entities (list): A list of entities, where each entity is a dictionary
            containing the entity text and its type.

    Returns:
        list: A list of EntitySpan objects, where each contains the entity
            type, start position, and end position.

    """
    result = []
    seen: DefaultDict[Tuple, int] = defaultdict(int)
    for entity in entities:
        entity_text = entity["entity_text"]
        entity_type = entity["entity_type"]
        if (entity_text, entity_type) in seen:
            continue

        matches = re.finditer(re.escape(entity_text), text, re.IGNORECASE)
        if matches:
            result.extend(
                [
                    EntitySpan(entity_type, match.start(), match.end())
                    for match in matches
                ]
            )
            seen[(entity_text, entity_type)] += 1

    return result


def fix_overlaps(spans: List[EntitySpan]) -> List[EntitySpan]:
    """Handle overlaps in entity spans"""

    if not spans:
        return spans

    # Sort spans and iterate to check for overlap
    sorted_spans = sorted(spans, key=lambda x: (x.start_position, -x.end_position))
    results = [sorted_spans[0]]

    for span in sorted_spans[1:]:
        if span.start_position < results[-1].end_position:
            # Skip candidate span if sub-interval (regardless of category)
            if span.end_position < results[-1].end_position:
                continue

            else:
                if span.entity_type == results[-1].entity_type:
                    # If overlapping spans have same type, then extend boundary
                    results[-1].end_position = span.end_position

                else:
                    # If overlapping spans have different type, add new span
                    results.append(
                        EntitySpan(
                            span.entity_type,
                            results[-1].end_position + 1,
                            span.end_position,
                        )
                    )
        else:
            # Add non-overlapping span
            results.append(span)

    return results
