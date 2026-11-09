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

"""Pipeline summaries and goal matching for prior runs."""

from __future__ import annotations

import contextlib
import re
from typing import TYPE_CHECKING, Any

from nemo_curator.stages.audio.audio_agent._safety import redact

if TYPE_CHECKING:
    from nemo_curator.stages.audio.audio_agent.recipe import Recipe


# Tokens shorter than this add noise to prompt↔summary matching (a, to, of, …).
_MIN_MATCH_TOKEN_LEN = 2


# Path-like params identify WHERE, not WHAT the pipeline did, and two runs of the same recipe
# on the same folder differ only in their scratch dirs — that is noise when comparing intent.
_SUMMARY_PATH_SUFFIXES = ("_dir", "_path", "_uri", "_file", "_folder")


_SUMMARY_PATH_KEYS = frozenset(
    {
        "output_path",
        "data_dir",
        "raw_data_dir",
        "resampled_audio_dir",
        "audio_dir",
        "manifest_filepath",
    }
)


# Written where a run has no recorded goal. Matched against as a literal so its filler words
# ("run", "that", "for") never count as intent overlap.
_NO_OBJECTIVE = "(objective not recorded for that run)"


SUMMARIZE_DIRECTIVE = (
    "pipeline_summary is the complete stage list with every behavioural param -- it is written "
    "for comparison, not for reading aloud. Never paste it verbatim. Retell it in one plain line: "
    "what the run produced, plus only the settings that differ between the priors you are showing "
    "or that the user asked about (thresholds, rates, channels, filters). Drop plumbing params."
)


_HOST_DIRECTIVE_FOLDER_RUNS = (
    "Before inventing a recipe, compare the user's current request to each prior's "
    "prompt (recorded goal) and pipeline_summary. Prefer a prior whose capabilities "
    "cover the full request over one that only covers a subset. Show the top 2–3 with "  # noqa: RUF001
    "stats; on pick use delta-run --from-run <run_id>. Do not invent a competing recipe first. " + SUMMARIZE_DIRECTIVE
)


def summarize_pipeline(recipe: Recipe | None) -> str:
    """What the run did: every stage with the params that decide its behaviour.

    Written onto a successful run so a later session can compare a new request to "what that
    run did" without loading the full recipe. Complete and uncapped on purpose -- truncating
    here is guesswork about which threshold matters, and a clipped line makes two runs that
    differ only in that threshold look identical. Condensing for display is the host's job;
    ``host_directive`` on the payload says so. Locations are dropped (see the constants above).
    """
    if recipe is None:
        return ""
    parts: list[str] = []
    for stage in recipe.stages:
        params = _summary_params(stage.semantic_params())
        if params:
            parts.append(f"{stage.ref}({', '.join(f'{k}={v}' for k, v in params)})")
        else:
            parts.append(stage.ref)
    return " -> ".join(parts)


def _summary_params(params: dict[str, Any]) -> list[tuple[str, Any]]:
    """Behavioural params in recipe order: everything bar locations and no-op settings."""
    safe_params = redact(params, redact_transcripts=False)
    return [(k, v) for k, v in safe_params.items() if not _is_location_param(k, v) and not _is_no_constraint(k, v)]


def _is_no_constraint(key: str, value: Any) -> bool:  # noqa: ANN401
    """A cap that caps nothing says nothing about the run -- ``max_samples=-1`` is not a choice."""
    if value is None:
        return True
    key_l = key.lower()
    unbounded_limit = key_l.endswith(("samples", "rows", "limit", "count"))
    return bool(unbounded_limit and isinstance(value, int) and value <= 0)


def _is_location_param(key: str, value: Any) -> bool:  # noqa: ANN401
    key_l = key.lower()
    if key_l in _SUMMARY_PATH_KEYS:
        return True
    if any(key_l.endswith(suffix) for suffix in _SUMMARY_PATH_SUFFIXES):
        return True
    return isinstance(value, str) and (value.startswith(("/", "~")))


def _match_aliases(key: str, value: Any) -> list[str]:  # noqa: ANN401
    """Words a person would use for a numeric param, added to the match corpus only.

    People ask for "16 kHz mono", recipes store ``target_sample_rate=16000`` and
    ``target_channels=1``. The aliases live here rather than in the displayed summary so
    matching does not push machine spellings like ``16000/16kHz`` in front of the reader.
    """
    key_l = key.lower()
    if "sample_rate" in key_l and isinstance(value, (int, float)) and value >= 1000:  # noqa: PLR2004
        rate = int(value)
        khz = rate // 1000
        return [str(rate), str(khz), f"{khz}k", f"{khz}khz"]
    if "channel" in key_l and isinstance(value, int):
        return {1: ["mono"], 2: ["stereo"]}.get(value, [])
    return []


def resolve_pipeline_summary(record: Any, recipe: Recipe | None = None) -> str:  # noqa: ANN401
    """Stored summary when present; otherwise derive from the record's recipe (older runs)."""
    stored = str(getattr(record, "pipeline_summary", "") or "").strip()
    if stored:
        return stored
    if recipe is not None:
        return summarize_pipeline(recipe)
    raw = getattr(record, "recipe", None)
    if not isinstance(raw, dict):
        return ""
    from nemo_curator.stages.audio.audio_agent.recipe import Recipe as _Recipe

    try:
        return summarize_pipeline(_Recipe.from_dict(raw))
    except ValueError:
        return ""


def goal_text(goal: dict[str, Any] | str | None) -> str:
    """Normalize a current or stored goal into a single comparable string."""
    if goal is None:
        return ""
    if isinstance(goal, str):
        return goal.strip()
    if isinstance(goal, dict):
        for key in ("task", "objective", "request", "summary"):
            if goal.get(key):
                return str(goal[key]).strip()
        return str(goal).strip() if goal else ""
    return str(goal).strip()


def prompt_summary_match(
    current_goal: dict[str, Any] | str | None,
    prior_prompt: str,
    pipeline_summary: str,
    *,
    match_corpus: str | None = None,
) -> dict[str, Any]:
    """How much of the current request is covered by the prior's prompt + pipeline.

    ``pipeline_summary`` is the short display string; ``match_corpus`` (when provided) is the
    uncapped stage+param text used for scoring so a truncated one-liner cannot hide a filter
    that actually ran. Score is coverage of current-request tokens. Empty current goal → 0.
    """
    current = _match_tokens(goal_text(current_goal))
    if match_corpus is None:
        prompt = "" if prior_prompt.strip() == _NO_OBJECTIVE else prior_prompt
        match_corpus = f"{prompt} {pipeline_summary}"
    prior = _match_tokens(match_corpus)
    if not current:
        return {"score": 0.0, "matched": [], "unmatched": [], "basis": "prior_prompt+pipeline_summary"}
    matched = sorted(current & prior)
    unmatched = sorted(current - prior)
    return {
        "score": round(len(matched) / len(current), 3),
        "matched": matched,
        "unmatched": unmatched,
        "basis": "prior_prompt+pipeline_summary",
    }


def match_corpus_for_record(record: Any, recipe: Recipe | None = None) -> str:  # noqa: ANN401
    """Uncapped prompt + stage refs + identifying params for ranking (not for display)."""
    from nemo_curator.stages.audio.audio_agent import reuse

    # A run with no recorded goal must score on its pipeline alone. Left in, the placeholder's
    # filler ("run", "that", "for") counts as intent overlap with almost any request.
    prompt = reuse._objective(record)
    if prompt.strip() == _NO_OBJECTIVE:
        prompt = ""
    if recipe is None:
        raw = getattr(record, "recipe", None)
        if isinstance(raw, dict):
            from nemo_curator.stages.audio.audio_agent.recipe import Recipe as _Recipe

            with contextlib.suppress(ValueError):
                recipe = _Recipe.from_dict(raw)
    parts = [prompt, resolve_pipeline_summary(record, recipe)]
    if recipe is not None:
        for stage in recipe.stages:
            parts.append(stage.ref)
            for key, value in _summary_params(stage.semantic_params()):
                parts.append(f"{key}={value}")
                parts.extend(_match_aliases(key, value))
    return " ".join(p for p in parts if p)


def _match_tokens(text: str) -> set[str]:
    """Alphanumeric tokens plus CamelCase splits so ``UTMOSFilterStage`` yields ``filter``."""
    lower = text.lower()
    tokens = {t for t in re.findall(r"[a-z0-9]+", lower) if len(t) >= _MIN_MATCH_TOKEN_LEN}
    for piece in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", text):
        p = piece.lower()
        if len(p) >= _MIN_MATCH_TOKEN_LEN:
            tokens.add(p)
    return tokens
