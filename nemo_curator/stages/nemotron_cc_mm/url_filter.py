"""Source-URL filters that operate on the metadata row of each sample."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from nemo_curator.stages.nemotron_cc_mm.text_filters import BaseInterleavedSampleFilterStage

if TYPE_CHECKING:
    pass


# Default substring blocklist applied to the source URL of each document.
# Conservative: high-precision tokens that almost never appear in benign URLs.
DEFAULT_NSFW_URL_SUBSTRINGS: tuple[str, ...] = (
    "porn", "xxx", "nsfw", "sex", "adult", "xnxx", "xvideos",
    "rule34", "hentai", "fetish", "cam-girl", "camgirl", "camsex",
)


def _get_source_url(group: pd.DataFrame) -> str | None:
    """Extract ``source_url`` from the sample's metadata row."""
    md = group[group["modality"] == "metadata"]
    if md.empty:
        return None
    src_ref = md.iloc[0]["source_ref"]
    if not isinstance(src_ref, str) or not src_ref:
        return None
    try:
        return json.loads(src_ref).get("source_url")
    except (ValueError, AttributeError):
        return None


@dataclass
class InterleavedURLSubstringNSFWFilterStage(BaseInterleavedSampleFilterStage):
    """Drop docs whose ``source_url`` contains any blocklisted substring.

    Cheap, high-precision safety pre-filter.  Operates only on the
    document URL recorded in the metadata row's ``source_ref``.
    """

    substrings: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_NSFW_URL_SUBSTRINGS
    )
    name: str = "interleaved_url_substring_nsfw_filter"

    def is_sample_ok(self, sample_id: str, group: pd.DataFrame) -> bool:
        url = _get_source_url(group)
        if not url:
            return True  # no URL → can't decide → keep
        url_lower = url.lower()
        return not any(s in url_lower for s in self.substrings)
