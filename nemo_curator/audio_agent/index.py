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

"""Knowledge Index — the deterministic retrieval backbone the host router queries.

Builds the category tree over the discovery catalog + capability cards, serves
tiered summaries (L0 categories -> L1 card one-liners -> L2 full cards) so the
host LLM can route coarse-to-fine without reading every card, and computes the
role-graph neighborhood used for composition and repair.

It provides material; it never makes the final relevance choice (that is the
host's job) and it never dumps source. All knowledge is static, versioned YAML
under ``knowledge/`` and ``recipes/`` — no learning, no run history.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_KNOWLEDGE_DIR = os.path.join(_HERE, "knowledge")
_RECIPES_DIR = os.path.join(_HERE, "recipes")

# Fallback category assignment when a stage has no card (keyword over module/class).
# (substring, category) — first match wins; order matters (specific before generic).
_MODULE_CATEGORY_RULES: tuple[tuple[str, str], ...] = (
    ("datasets", "ingest"),
    ("ManifestReader", "ingest"),
    ("ManifestWriter", "export"),
    ("resample", "preprocess"),
    ("mono_conversion", "preprocess"),
    ("concatenation", "segment"),
    ("segmentation.vad", "segment"),
    ("inference.vad", "segment"),
    ("tagging.split", "segment"),
    ("speaker_diarization", "diarize"),
    ("speaker_separation", "diarize"),
    ("inference.asr", "transcribe"),
    ("nemo_asr_align", "transcribe"),
    ("merge_alignment", "transcribe"),
    ("tagging.text", "text_norm"),
    ("metrics", "quality"),
    ("filtering", "quality"),
    ("PreserveByValue", "filter"),
    ("alm", "alm"),
    ("io.", "export"),
    ("postprocessing", "export"),
    ("prepare_module_segments", "export"),
    ("extract_segments", "export"),
)


def _load_yaml(path: str) -> Any:  # noqa: ANN401
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_dir(path: str) -> dict[str, Any]:
    """Load every ``*.yaml`` in a directory into ``{stem: parsed}`` (missing dir -> {})."""
    out: dict[str, Any] = {}
    if not os.path.isdir(path):
        return out
    for fn in sorted(os.listdir(path)):
        if fn.endswith((".yaml", ".yml")):
            stem = os.path.splitext(fn)[0]
            try:
                out[stem] = _load_yaml(os.path.join(path, fn))
            except Exception:  # noqa: BLE001, S112 - a malformed knowledge file must not break discovery
                continue
    return out


class KnowledgeIndex:
    """Loads knowledge + catalog and answers tiered retrieval / role queries."""

    def __init__(self) -> None:
        self._cards = _load_dir(os.path.join(_KNOWLEDGE_DIR, "cards"))
        self._blueprints = _load_dir(os.path.join(_KNOWLEDGE_DIR, "blueprints"))
        self._patterns = _load_dir(os.path.join(_KNOWLEDGE_DIR, "patterns"))
        self._recipes = _load_dir(_RECIPES_DIR)
        taxonomy_path = os.path.join(_KNOWLEDGE_DIR, "taxonomy.yaml")
        self._taxonomy = _load_yaml(taxonomy_path) if os.path.isfile(taxonomy_path) else {}
        # index cards by their stage_id (fall back to file stem)
        self._cards_by_stage: dict[str, dict[str, Any]] = {}
        for stem, card in self._cards.items():
            if isinstance(card, dict):
                self._cards_by_stage[str(card.get("stage_id", stem))] = card

    # ------------------------------------------------------------------ #
    # catalog + categories
    # ------------------------------------------------------------------ #
    def stage_names(self) -> list[str]:
        from nemo_curator.stages.audio._agent._catalog import list_agent_ready_stages

        return list_agent_ready_stages()

    def card(self, stage: str) -> dict[str, Any] | None:
        return self._cards_by_stage.get(stage)

    def all_cards(self) -> dict[str, dict[str, Any]]:
        """Every loaded card keyed by ``stage_id`` (used by the card conformance gate)."""
        return dict(self._cards_by_stage)

    def category_of(self, stage: str) -> str:
        card = self.card(stage)
        if card and card.get("category"):
            return str(card["category"])
        return self._default_category(stage)

    def _default_category(self, stage: str) -> str:
        try:
            from nemo_curator.audio_agent._resolve import resolve_stage_class

            cls = resolve_stage_class(stage)
            hay = f"{cls.__module__}.{cls.__name__}"
        except Exception:  # noqa: BLE001
            hay = stage
        for needle, cat in _MODULE_CATEGORY_RULES:
            if needle in hay:
                return cat
        return "other"

    def one_liner(self, stage: str) -> str:
        card = self.card(stage)
        if card and card.get("summary"):
            return str(card["summary"])
        try:
            from nemo_curator.audio_agent._resolve import static_contract_for

            c = static_contract_for(stage)
            return c.description or stage  # noqa: TRY300
        except Exception:  # noqa: BLE001
            return stage

    def tags_of(self, stage: str) -> list[str]:
        card = self.card(stage)
        tags = list(card.get("tags", [])) if card else []
        return [str(t) for t in tags]

    def category_tree(self) -> list[dict[str, Any]]:
        """L0: the full (small) list of categories with descriptions + member stages."""
        buckets: dict[str, list[str]] = {}
        for name in self.stage_names():
            buckets.setdefault(self.category_of(name), []).append(name)
        tax = self._taxonomy.get("categories", {}) if isinstance(self._taxonomy, dict) else {}
        order = list(tax.keys())
        tree: list[dict[str, Any]] = []
        for cat in sorted(buckets, key=lambda c: (order.index(c) if c in order else 999, c)):
            desc = ""
            if isinstance(tax.get(cat), dict):
                desc = str(tax[cat].get("description", ""))
            elif isinstance(tax.get(cat), str):
                desc = tax[cat]
            tree.append({"category": cat, "description": desc, "stages": sorted(buckets[cat])})
        return tree

    def card_oneliners(self, category: str) -> list[dict[str, str]]:
        """L1: one-liners for the stages within a chosen category."""
        return [
            {"stage": name, "summary": self.one_liner(name), "tags": self.tags_of(name)}
            for name in sorted(self.stage_names())
            if self.category_of(name) == category
        ]

    def full_cards(self, stages: list[str]) -> list[dict[str, Any]]:
        """L2: full card + static contract for the finalists the router selected."""
        out: list[dict[str, Any]] = []
        for name in stages:
            entry: dict[str, Any] = {"stage": name, "category": self.category_of(name)}
            card = self.card(name)
            if card:
                entry["card"] = card
            try:
                from nemo_curator.audio_agent._resolve import static_contract_for

                entry["contract"] = static_contract_for(name).to_dict()
            except Exception as e:  # noqa: BLE001 - stage may need an optional dep to describe
                entry["contract_error"] = f"{type(e).__name__}: {e}"
            out.append(entry)
        return out

    # ------------------------------------------------------------------ #
    # blueprints / patterns / recipes (retrieval by simple token overlap)
    # ------------------------------------------------------------------ #
    def blueprints(self) -> list[dict[str, Any]]:
        return [b for b in self._blueprints.values() if isinstance(b, dict)]

    def patterns(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for v in self._patterns.values():
            if isinstance(v, dict) and "patterns" in v and isinstance(v["patterns"], list):
                out.extend(p for p in v["patterns"] if isinstance(p, dict))
            elif isinstance(v, list):
                out.extend(p for p in v if isinstance(p, dict))
            elif isinstance(v, dict):
                out.append(v)
        return out

    def recipes(self) -> list[dict[str, Any]]:
        return [r for r in self._recipes.values() if isinstance(r, dict)]

    def match_blueprints(self, goal: dict[str, Any], k: int = 3) -> list[dict[str, Any]]:
        return self._top_k(self.blueprints(), goal, k, fields=("blueprint_id", "intent", "data_assumptions"))

    def match_recipes(self, goal: dict[str, Any], k: int = 3) -> list[dict[str, Any]]:
        return self._top_k(self.recipes(), goal, k, fields=("recipe_id", "name", "intent", "rationale"))

    @staticmethod
    def _top_k(
        items: list[dict[str, Any]], goal: dict[str, Any], k: int, fields: tuple[str, ...]
    ) -> list[dict[str, Any]]:
        """Entries whose text overlaps the goal, best first; EMPTY when none overlap.

        Returning an arbitrary few on no match was actively misleading: these are presented
        as ``matched_blueprints``/``matched_recipes`` and their presets are pulled into the
        planning context, so a Japanese diarization goal was handed the three bundled
        English read-speech examples -- complete with an English-Parakeet preset -- as
        though they had been selected for it. No match is a real answer: the host composes
        from the catalog instead, which the skill already describes.

        A goal with no usable tokens is different: nothing was asked, so offering what
        exists is a browse, not a claim of relevance.
        """
        goal_tokens = _tokens(" ".join(str(v) for v in goal.values()))
        if not goal_tokens:
            return items[:k]
        scored: list[tuple[int, dict[str, Any]]] = []
        for it in items:
            text = " ".join(str(it.get(f, "")) for f in fields)
            scored.append((len(goal_tokens & _tokens(text)), it))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [it for score, it in scored if score > 0][:k]

    # ------------------------------------------------------------------ #
    # role graph
    # ------------------------------------------------------------------ #
    def role_neighborhood(self, roles: list[str] | None = None) -> dict[str, Any]:
        from nemo_curator.stages.audio._agent._catalog import role_index

        idx = role_index()
        if roles is None:
            return idx
        rset = set(roles)
        return {
            "producers": {r: idx["producers"].get(r, []) for r in rset},
            "consumers": {r: idx["consumers"].get(r, []) for r in rset},
            "unresolved_stages": idx.get("unresolved_stages", []),
        }

    def unproducible(self, roles: list[str]) -> list[str]:
        from nemo_curator.stages.audio._agent._catalog import find_producers

        return sorted({r for r in roles if r != "unknown" and not find_producers(r)})


def _tokens(text: str) -> set[str]:
    return {t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if len(t) > 2}  # noqa: PLR2004


@lru_cache(maxsize=1)
def get_index() -> KnowledgeIndex:
    """Process-wide cached Knowledge Index (knowledge YAML is static)."""
    return KnowledgeIndex()
