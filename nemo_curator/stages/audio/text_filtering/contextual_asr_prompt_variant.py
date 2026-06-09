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

"""Contextual ASR prompt variant stage (CPU-only).

Reads the entity extraction dict produced by
:class:`ContextualASRExtractionStage` and appends five prompt-variant
fields to it (contextless prompts are handled separately).  Each
variant provides a different level of context information for training
a context-biased ASR model.

The stage is language-aware: when a per-sample source language is
available under ``source_lang_key`` (default ``source_lang``), about half
of the templates sampled per sample explicitly mention the language
(e.g. ``"Transcribe the French audio …"``).  Templates that don't
reference the language remain in the pool too, so the rendered prompts
still vary in phrasing.  When the source-lang field is missing or
empty, the stage falls back to the language-agnostic templates only,
preserving back-compat with older manifests.

This stage is deterministic — it uses a per-sample seeded RNG derived
from ``sha256(seed + audio_filepath)`` so results are reproducible
regardless of batch order, worker count, or shard layout.  The domain
pool used for negative sampling is fixed at init and never mutated.
No GPU or LLM is required.
"""

from __future__ import annotations

import hashlib
import random as _random_module
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

# ─────────────────────────────────────────────────────────────
#  Predefined domain pool for negative-context sampling
# ─────────────────────────────────────────────────────────────

_DEFAULT_DOMAIN_POOL: list[str] = [
    "Automotive Engineering",
    "Aviation and Aerospace",
    "Climate Science",
    "Consumer Electronics",
    "Criminal Justice",
    "Cryptocurrency and Blockchain",
    "Culinary Arts",
    "Cybersecurity",
    "Daily Conversation",
    "Education and Pedagogy",
    "Environmental Policy",
    "Fashion and Apparel",
    "Film and Television",
    "Financial Markets",
    "Fitness and Sports Medicine",
    "Gaming and Esports",
    "Healthcare and Medicine",
    "Immigration Law",
    "Marine Biology",
    "Music Production",
    "Nuclear Physics",
    "Pharmaceutical Research",
    "Photography and Visual Arts",
    "Political Science",
    "Real Estate",
    "Renewable Energy",
    "Robotics and Automation",
    "Software Engineering",
    "Space Exploration",
    "Veterinary Medicine",
]

# ─────────────────────────────────────────────────────────────
#  Prompt template libraries
# ─────────────────────────────────────────────────────────────

COARSE_TEMPLATES: list[str] = [
    "This audio belongs to the {domain} field. Transcribe the audio into text, ensuring all punctuation marks are included.",
    "Domain: {domain}. Transcribe the audio accurately.",
    "This is a recording related to {domain}. Write out what is being said.",
    "The following audio is from the {domain} domain. Please transcribe it.",
    "Topic: {domain}. Listen and produce an accurate transcript.",
    "Context: this audio discusses {domain}. Transcribe the spoken content.",
    "Transcribe the following audio. For context, the recording is about {domain}.",
    "Please transcribe this audio recording. It is related to the field of {domain}.",
    "Write out what is said in this audio. The topic is {domain}.",
    "Produce a transcript of the audio below. The subject area is {domain}.",
    "You are transcribing a recording from the {domain} field. Pay attention to domain-specific terminology. Transcribe accurately.",
    "As an expert in {domain}, transcribe the following audio. Use appropriate terminology for the field.",
    "The speaker is discussing a topic in {domain}. Transcribe their words precisely.",
    "Field: {domain}. Transcribe.",
    "Audio topic: {domain}. Provide the transcript.",
]

# Language-aware coarse templates.
COARSE_LANG_TEMPLATES: list[str] = [
    "This {language} audio belongs to the {domain} field. Transcribe the audio into text, ensuring all punctuation marks are included.",
    "Domain: {domain}. Transcribe the {language} audio accurately.",
    "You are transcribing a {language} recording from the {domain} field. Pay attention to domain-specific terminology.",
    "Topic: {domain}. Listen to the {language} audio and produce an accurate transcript.",
    "The following {language} audio is from the {domain} domain. Please transcribe it.",
    "Field: {domain} ({language}). Transcribe.",
    "As an expert in {domain}, transcribe the following {language} audio. Use appropriate terminology for the field.",
]

FINE_TEMPLATES: list[str] = [
    "This audio belongs to the {domain} field and may contain the following words or phrases: {entities}. Transcribe the audio into text, ensuring all punctuation marks are included.",
    "Domain: {domain}. Key terms that may appear: {entities}. Transcribe the audio accurately.",
    "This recording is about {domain}. The audio may mention: {entities}. Please transcribe.",
    "Topic: {domain}. Expected vocabulary: {entities}. Write out what is spoken.",
    "The following audio is from the {domain} field. It may reference these terms: {entities}. Produce an accurate transcript.",
    "This audio discusses {domain} and may include terms such as {entities}. Transcribe what you hear.",
    "You are transcribing audio about {domain}. Pay special attention to these terms if they appear: {entities}.",
    "The speaker is talking about {domain}. Words and phrases to watch for: {entities}. Transcribe precisely.",
    "Transcribe this audio. For reference, the topic is {domain} and the following terms may be mentioned: {entities}.",
    "Please transcribe the following recording. Contextual hint — the domain is {domain}, and relevant terms include: {entities}.",
    "Listen and transcribe the audio. The recording is from the {domain} field. Terminology that may appear: {entities}.",
    "You are an expert transcriptionist specializing in {domain}. The audio may contain the following specialized terms: {entities}. Produce a precise transcript.",
    "Transcribe the audio below. The recording covers {domain}. Be especially careful with these terms, which may appear in the audio: {entities}.",
    "Field: {domain}. Vocabulary hints: {entities}. Transcribe.",
    "Topic: {domain}. Terms: {entities}. Write the transcript.",
]

# Language-aware fine templates.
FINE_LANG_TEMPLATES: list[str] = [
    "This {language} audio belongs to the {domain} field and may contain the following words or phrases: {entities}. Transcribe the audio into text, ensuring all punctuation marks are included.",
    "Domain: {domain}. Key terms that may appear in the {language} audio: {entities}. Transcribe accurately.",
    "This {language} recording is about {domain} and may include terms such as {entities}. Transcribe what you hear.",
    "You are transcribing {language} audio about {domain}. Pay special attention to these terms if they appear: {entities}.",
    "Transcribe this {language} audio. For reference, the topic is {domain} and the following terms may be mentioned: {entities}.",
    "You are an expert {language} transcriptionist specializing in {domain}. The audio may contain the following specialized terms: {entities}.",
    "Field: {domain} ({language}). Vocabulary hints: {entities}. Transcribe.",
]

_ENTITY_STYLES: list[str] = ["comma", "and", "semicolon", "quoted", "numbered"]


# ─────────────────────────────────────────────────────────────
#  Formatting helpers
# ─────────────────────────────────────────────────────────────


def _format_entity_list(entities: list[str], style: str = "comma") -> str:  # noqa: PLR0911
    if not entities:
        return ""
    if style == "comma":
        return ", ".join(entities)
    if style == "and":
        if len(entities) == 1:
            return entities[0]
        return ", ".join(entities[:-1]) + ", and " + entities[-1]
    if style == "semicolon":
        return "; ".join(entities)
    if style == "quoted":
        return ", ".join(f'"{e}"' for e in entities)
    if style == "numbered":
        return "; ".join(f"({i + 1}) {e}" for i, e in enumerate(entities))
    return ", ".join(entities)


# ─────────────────────────────────────────────────────────────
#  Per-variant pickers
# ─────────────────────────────────────────────────────────────


def _pick_coarse(
    domain: str,
    rng: _random_module.Random,
    language: str | None = None,
) -> str:
    pool = COARSE_TEMPLATES + (COARSE_LANG_TEMPLATES if language else [])
    template = rng.choice(pool)
    fmt: dict[str, str] = {"domain": domain}
    if language and "{language}" in template:
        fmt["language"] = language
    return template.format(**fmt)


def _pick_fine(
    domain: str,
    entities: list[str],
    rng: _random_module.Random,
    language: str | None = None,
) -> str:
    pool = FINE_TEMPLATES + (FINE_LANG_TEMPLATES if language else [])
    template = rng.choice(pool)
    style = rng.choice(_ENTITY_STYLES)
    entity_str = _format_entity_list(entities, style)
    fmt: dict[str, str] = {"domain": domain, "entities": entity_str}
    if language and "{language}" in template:
        fmt["language"] = language
    return template.format(**fmt)


def _pick_distractor(
    domain: str,
    fine_entities: list[str],
    distractors: list[str],
    rng: _random_module.Random,
    language: str | None = None,
) -> str:
    combined = list(fine_entities) + list(distractors)
    rng.shuffle(combined)
    return _pick_fine(domain, combined, rng, language=language)


def _pick_partial(  # noqa: PLR0913
    domain: str,
    entities: list[str],
    rng: _random_module.Random,
    keep_lo: float = 0.5,
    keep_hi: float = 0.8,
    language: str | None = None,
) -> str:
    frac = rng.uniform(keep_lo, keep_hi)
    k = max(1, int(len(entities) * frac))
    subset = rng.sample(entities, min(k, len(entities)))
    return _pick_fine(domain, subset, rng, language=language)


def _pick_negative(
    wrong_domain: str,
    entities: list[str] | None,
    rng: _random_module.Random,
    language: str | None = None,
) -> str:
    if entities:
        return _pick_fine(wrong_domain, entities, rng, language=language)
    return _pick_coarse(wrong_domain, rng, language=language)


def _render_all_variants(  # noqa: PLR0913
    extraction: dict,
    domain_pool: list[str],
    rng: _random_module.Random,
    partial_keep_lo: float = 0.5,
    partial_keep_hi: float = 0.8,
    min_entities_for_partial: int = 3,
    language: str | None = None,
) -> dict[str, str]:
    coarse_terms = extraction.get("coarse_context_terms") or []
    fine_terms = extraction.get("fine_context_terms") or []
    distractor_terms = extraction.get("distractor_terms") or []
    domain = coarse_terms[0] if coarse_terms else "General"

    coarse_context_prompt = _pick_coarse(domain, rng, language=language)

    fine_context_prompt = (
        _pick_fine(domain, fine_terms, rng, language=language) if fine_terms else coarse_context_prompt
    )
    distractor_prompt = (
        _pick_distractor(domain, fine_terms, distractor_terms, rng, language=language)
        if fine_terms or distractor_terms
        else coarse_context_prompt
    )
    partial_context_prompt = (
        _pick_partial(
            domain,
            fine_terms,
            rng,
            keep_lo=partial_keep_lo,
            keep_hi=partial_keep_hi,
            language=language,
        )
        if len(fine_terms) >= min_entities_for_partial
        else fine_context_prompt
    )

    excluded = set(coarse_terms)
    other_domains = [d for d in domain_pool if d and d not in excluded]
    wrong_domain = rng.choice(other_domains) if other_domains else "Unrelated Topic"
    negative_entities = fine_terms[:3] if fine_terms else None
    negative_context_prompt = _pick_negative(
        wrong_domain,
        negative_entities,
        rng,
        language=language,
    )

    return {
        "coarse_context_prompt": coarse_context_prompt,
        "fine_context_prompt": fine_context_prompt,
        "distractor_prompt": distractor_prompt,
        "partial_context_prompt": partial_context_prompt,
        "negative_context_prompt": negative_context_prompt,
    }


# ─────────────────────────────────────────────────────────────
#  Stage
# ─────────────────────────────────────────────────────────────


@dataclass
class ContextualASRPromptVariantStage(ProcessingStage[AudioTask, AudioTask]):
    """Generate five prompt variants from extracted context-ASR entities.

    Reads the extraction dict written by
    :class:`ContextualASRExtractionStage` (under ``context_key``) and
    appends five prompt-variant fields to the same dict.

    This is a CPU-only stage — no LLM or GPU required.

    The five variants provide different levels of context information
    for training a context-biased ASR model:

    - ``coarse_context_prompt``: domain label only
    - ``fine_context_prompt``: domain + entity list
    - ``distractor_prompt``: domain + entities shuffled with distractors
    - ``partial_context_prompt``: domain + random 50-80% of entities
    - ``negative_context_prompt``: wrong-domain label

    Args:
        context_key: Manifest key holding the extraction dict.
        sample_id_key: Manifest key used to derive the per-sample RNG
            seed.  Must be a stable identifier (e.g. ``audio_filepath``)
            so that outputs are deterministic regardless of batch order,
            worker count, or shard layout.
        source_lang_key: Manifest key holding the per-sample source
            language (display name or code).  When present and non-empty
            the stage mixes language-aware templates into the sampling
            pool.  When missing or empty, the stage falls back to the
            language-agnostic templates only.
        seed: Base RNG seed combined with the sample id hash.
        partial_keep_lo: Lower bound of the random keep fraction for the
            partial-context variant (default 0.5).
        partial_keep_hi: Upper bound of the random keep fraction for the
            partial-context variant (default 0.8).  Each sample draws a
            uniform fraction in ``[partial_keep_lo, partial_keep_hi]``.
        min_entities_for_partial: Minimum entity count to produce a
            distinct partial variant (below this, falls back to fine).
        domain_pool: Optional list of domain labels for negative
            sampling.  Uses a built-in default pool if not provided.
            The pool is never mutated at runtime.
        num_workers_override: Explicit worker count for Xenna.
    """

    name: str = "ContextualASRPromptVariant"
    context_key: str = "context_asr"
    sample_id_key: str = "audio_filepath"
    source_lang_key: str = "source_lang"
    seed: int = 42
    partial_keep_lo: float = 0.5
    partial_keep_hi: float = 0.8
    min_entities_for_partial: int = 3
    domain_pool: list[str] | None = None
    num_workers_override: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 256

    _pool: list[str] = field(default_factory=list, init=False, repr=False)

    def num_workers(self) -> int | None:
        return self.num_workers_override

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.num_workers_override is not None:
            spec["num_workers"] = self.num_workers_override
        return spec

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        self._pool = list(self.domain_pool) if self.domain_pool else list(_DEFAULT_DOMAIN_POOL)

    def teardown(self) -> None:
        pass

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.context_key, self.source_lang_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.context_key]

    def _sample_seed(self, task: AudioTask) -> int:
        sample_id = task.data.get(self.sample_id_key, "")
        h = hashlib.sha256(f"{self.seed}:{sample_id}".encode()).digest()
        return int.from_bytes(h[:8], "little")

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []

        for task in tasks:
            extraction = task.data.get(self.context_key)
            if not isinstance(extraction, dict):
                continue

            language = task.data.get(self.source_lang_key) or None
            language = str(language) if language else None
            rng = _random_module.Random(self._sample_seed(task))  # noqa: S311
            variants = _render_all_variants(
                extraction,
                domain_pool=self._pool,
                rng=rng,
                partial_keep_lo=self.partial_keep_lo,
                partial_keep_hi=self.partial_keep_hi,
                min_entities_for_partial=self.min_entities_for_partial,
                language=language,
            )
            extraction.update(variants)

        logger.debug("{}: batch of {} tasks", self.name, len(tasks))
        return tasks
