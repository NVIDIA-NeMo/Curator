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

import pytest

from nemo_curator.stages.base import _camel_to_snake_case, _validate_stage_name

SNAKE_CASE_PATTERN = re.compile(r"[a-z][a-z0-9_]*")


def test_validate_stage_name():
    """Test the stage name validation function."""

    # Test valid snake_case names
    valid_names = [
        "valid_name",
        "snake_case_name",
        "name_with_123",
        "simple",
        "a",
        "word_word_word",
        "deduplication_stage",
        "common_crawl_main_pipeline",
    ]

    for name in valid_names:
        _validate_stage_name(name)  # Should not raise

    # Test invalid names
    invalid_names = [
        "CamelCase",
        "PascalCase",
        "InvalidCamelCase",
        "RemovalStage",
        "PairwiseCosineSimilarityStage",
        "KMeansStage",
        "DuplicatesRemovalStage",
        "123invalid",
        "_starts_with_underscore",
        "has-dashes",
        "has spaces",
        "has.dots",
        "ALLCAPS",
        "",
        "Mixed_Case_Name",
    ]

    for name in invalid_names:
        with pytest.raises(ValueError, match="Stage name must be snake_case"):
            _validate_stage_name(name)


def test_camel_to_snake_case():
    """Test the CamelCase to snake_case conversion function."""
    test_cases = [
        ("ConnectedComponentsStage", "connected_components_stage"),
        ("MinHashStage", "min_hash_stage"),
        ("DocumentModifier", "document_modifier"),
        ("FilterFn", "filter_fn"),
        ("SimpleClass", "simple_class"),
        ("XMLParser", "xml_parser"),
        ("HTTPRequest", "http_request"),
        ("PDFReader", "pdf_reader"),
        ("HTMLExtractor", "html_extractor"),
        ("URLGenerator", "url_generator"),
    ]

    for camel, expected_snake in test_cases:
        result = _camel_to_snake_case(camel)
        assert result == expected_snake, f"Expected {camel} -> {expected_snake}, got {result}"


def test_stage_naming_convention():
    """Test that all stage instances have snake_case names."""

    # This is a general test that can be used to validate any stage instance
    def assert_stage_name_is_snake_case(stage_instance: object) -> None:
        """Assert that a stage instance has a snake_case name."""
        assert hasattr(stage_instance, "name"), f"Stage {stage_instance} must have a 'name' attribute"
        stage_name = stage_instance.name
        assert isinstance(stage_name, str), f"Stage name must be a string, got {type(stage_name)}"
        assert SNAKE_CASE_PATTERN.fullmatch(stage_name), f"Stage name '{stage_name}' must be snake_case"

    # This test can be extended by importing and testing specific stage instances
    # For now, just test that the test function itself works
    class MockStage:
        def __init__(self, name: str) -> None:
            self.name = name

    # Test valid stage
    valid_stage = MockStage("valid_snake_case_name")
    assert_stage_name_is_snake_case(valid_stage)

    # Test invalid stage
    invalid_stage = MockStage("InvalidCamelCase")
    with pytest.raises(AssertionError, match="must be snake_case"):
        assert_stage_name_is_snake_case(invalid_stage)


def test_class_names_are_camel_case():
    """Test that stage class names follow CamelCase convention."""

    # Test the naming convention function
    def assert_class_name_is_camel_case(cls: type) -> None:
        """Assert that a class name follows CamelCase convention."""
        class_name = cls.__name__
        # CamelCase starts with uppercase and contains only letters/digits
        assert re.match(r"^[A-Z][a-zA-Z0-9]*$", class_name), f"Class name '{class_name}' must be CamelCase"
        # Ensure it's not all uppercase (which would be SCREAMING_SNAKE_CASE)
        assert not class_name.isupper() or len(class_name) == 1, (
            f"Class name '{class_name}' should not be all uppercase"
        )

    # Test some example class names
    class ValidCamelCaseStage:
        pass

    class AnotherValidStage:
        pass

    assert_class_name_is_camel_case(ValidCamelCaseStage)
    assert_class_name_is_camel_case(AnotherValidStage)
