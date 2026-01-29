#!/usr/bin/env python3
"""Verify text modality dependencies are installed."""
from nemo_curator.stages.text.deduplication import TextDuplicatesRemovalWorkflow
from nemo_curator.stages.text.filters import WordCountFilter
print("✓ Text modality imports verified")
