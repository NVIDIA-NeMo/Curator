# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest

from nemo_curator.stages.interleaved.pdf.nemotron_parse.page_budget import (
    document_identity,
    output_version,
    selected_pages,
)
from tests.stages.interleaved.pdf.nemotron_parse._page_budget_helpers import TEST_EXTRACTION_CONFIG


def test_document_identity_is_independent_of_task_layout_and_manifest_metadata():
    source = {"source_locator": "s3://bucket/doc.pdf", "source_version": "sha256:immutable"}
    identity = document_identity({**source, "num_pages": 200, "pages_per_task": 64, "url": "old"})
    assert document_identity({**source, "num_pages": 50, "selected_page_indices": [8], "url": "new"}) == identity
    assert document_identity({**source, "source_version": "different"})["document_id"] != identity["document_id"]
    assert (
        document_identity({**source, "source_locator": "s3://other/doc.pdf"})["document_id"] != identity["document_id"]
    )


@pytest.mark.parametrize("field", ["source_locator", "source_version"])
@pytest.mark.parametrize("value", [None, "", " ", 42])
def test_unversioned_or_ambiguous_sources_are_rejected(field: str, value: object):
    entry = {"source_locator": "file:///corpus/a.pdf", "source_version": "v1", field: value}
    with pytest.raises(ValueError, match=field):
        document_identity(entry)


@pytest.mark.parametrize("count", [0, -1, True, 1.1, "100"])
def test_invalid_counts_are_rejected(count: object):
    with pytest.raises(ValueError, match="num_pages"):
        selected_pages({"num_pages": count})


@pytest.mark.parametrize("selection", [[], [1, 1], [-1], [True], [3]])
def test_invalid_or_out_of_range_selection_is_rejected(selection: list):
    with pytest.raises(ValueError, match="selected_page_indices"):
        selected_pages({"num_pages": 3, "selected_page_indices": selection})


def test_page_mode_selects_the_full_document_or_explicit_selection():
    assert selected_pages({"num_pages": 200}) == list(range(200))
    assert selected_pages({"num_pages": 100, "selected_page_indices": [80, 2, 5]}) == [80, 2, 5]
    assert selected_pages({"selected_page_indices": [80, 2, 5]}) == [80, 2, 5]
    with pytest.raises(ValueError, match="requires num_pages"):
        selected_pages({})


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("parser", "version", "new-parser"),
        ("parser", "source_sha256", "c" * 64),
        ("model", "revision", "new-model"),
        ("model", "artifacts_sha256", "d" * 64),
        ("request", "prompt", "changed prompt"),
        ("request", "generation", {"temperature": 0.8}),
        ("transport", "max_retries", 5),
    ],
)
def test_effective_extraction_changes_change_output_identity(section: str, key: str, value: str | dict):
    config = copy.deepcopy(TEST_EXTRACTION_CONFIG)
    previous = output_version({"pdf_extraction_config": config})
    config[section][key] = value
    assert output_version({"pdf_extraction_config": config}) != previous


def test_ignored_arguments_and_task_layout_do_not_change_extraction_identity():
    metadata = {"pdf_extraction_config": TEST_EXTRACTION_CONFIG}
    assert output_version(metadata) == output_version(
        {
            **metadata,
            "pages_per_task": 128,
            "model_path": "ignored alias",
            "pdf_parser_config": {"max_tokens": 100, "text_in_pic": True, "max_pages": 1},
        }
    )


def test_unresolved_extraction_identity_fails_closed():
    with pytest.raises(ValueError, match="resolved"):
        output_version({"pdf_parser_config": {"max_tokens": 8744}})
