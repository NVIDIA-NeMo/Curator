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

"""Instruction packing stage (CPU-only).

Walks the per-task output keys produced by the text post-processing
pipeline (``pnc_text``, ``tn_raw``, ``itn_raw``, ``itn_clean``,
``captioning_text``, ``code_switched_text``, ``speech_qa_text``, and
optionally the ``context_asr`` nested dict) and assembles a single
``preference_instructions`` field — a list of
``{"prompt": ..., "target": ..., "tags": {"type": ..., "target_lang": ...}}``
entries — that downstream trainers can sample from one-per-epoch.

The stage is deterministic: prompt-template selection uses a
per-sample seeded RNG derived from ``sha256(seed + audio_filepath)``
so output is reproducible regardless of worker count or batch order.
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
#  Prompt template libraries (one per task type)
# ─────────────────────────────────────────────────────────────

# Punctuation-agnostic prompts; shared by both tn and tn-pnc.
_GENERIC_TRANSCRIBE_PROMPTS: list[str] = [
    "Transcribe this audio.",
    "Write out what is being said in this audio clip.",
    "Convert the speech in this audio to text.",
    "Listen to the audio and produce an accurate transcript of what is spoken.",
    "Please transcribe the following audio recording word for word.",
    "Carefully transcribe the spoken content in this audio file.",
    "What is being said in this audio? Provide the full transcript.",
    "Produce a verbatim transcript of the audio.",
    "You are a professional transcriptionist. Transcribe the following audio recording accurately, capturing every word as spoken.",
    "Your task is to listen to this audio and write down exactly what is said, including all proper nouns and technical terms.",
    "Transcribe the audio below. Be precise with names, numbers, and specialized vocabulary.",
    "Listen carefully and transcribe the audio. Preserve the original phrasing and word choices.",
    "Generate an accurate text transcript of the speech in this recording.",
]

# Punctuation/capitalization-explicit; tn-pnc only.
_PNC_ONLY_PROMPTS: list[str] = [
    "Transcribe the audio with proper punctuation and capitalization.",
    "Provide a transcript of this audio including punctuation marks.",
    "Listen to the audio and write down what is said, using correct punctuation.",
    "Produce a transcript with sentence boundaries and capitalization.",
    "Write what is spoken in this clip, including commas, periods, and capital letters where appropriate.",
    "Transcribe the audio into text, ensuring all punctuation marks are included.",
    "Transcribe this audio. Include proper punctuation and capitalization.",
]

# No-punctuation/lowercase-explicit; tn only.
_TN_ONLY_PROMPTS: list[str] = [
    "Transcribe the audio as plain normalized text, without punctuation or capitalization.",
    "Write down the spoken words only — no punctuation marks, no capital letters.",
    "Produce a normalized transcript: lowercase words, no punctuation.",
]

_PNC_PROMPTS: list[str] = _GENERIC_TRANSCRIBE_PROMPTS + _PNC_ONLY_PROMPTS
_TN_PROMPTS: list[str] = _GENERIC_TRANSCRIBE_PROMPTS + _TN_ONLY_PROMPTS

_ITN_PROMPTS: list[str] = [
    "Transcribe the audio in written form: render numbers, currencies, and dates using digits and symbols.",
    "Produce a transcript with inverse text normalization — write '25' instead of 'twenty five', '$100' instead of 'a hundred dollars'.",
    "Transcribe what is said and convert spoken numbers, currencies, and units into their written-form equivalents.",
    "Write a transcript using digits, symbols, and standard abbreviations rather than spelled-out forms.",
]

_ITN_NO_DISFL_PROMPTS: list[str] = [
    "Transcribe the audio cleanly: remove filler words ('um', 'uh', 'you know'), repetitions, and false starts. Also apply inverse text normalization (digits, symbols).",
    "Produce a clean, readable transcript with disfluencies removed and numbers/units in written form.",
    "Transcribe what is said but drop all hesitation markers and repeated words; render numbers and currencies as digits.",
    "Write a polished transcript: strip out fillers and stutters, and use written-form numbers and symbols.",
]

_CAPTION_PROMPTS: list[str] = [
    "Provide a short caption that describes what this audio is about.",
    "Summarize the content of this audio in one short sentence.",
    "Generate a brief caption for this audio clip.",
    "Write a one-line summary of what is said in this recording.",
    "Describe in a few words the topic of this audio.",
]

_CODE_SWITCH_PROMPTS: list[str] = [
    "Transcribe the audio, preserving any English words in their original Latin spelling and the rest in the native script.",
    "Produce a code-switched transcript: native-language words in native script, English loanwords in standard English spelling.",
    "Transcribe what is said, keeping English code-switches in Latin script and native words in their own script.",
    "Write a transcript that reflects the real code-switched utterance — English words in English, native words in native script.",
]

_CONTEXT_ASR_VARIANT_KEYS: list[str] = [
    "coarse_context_prompt",
    "fine_context_prompt",
    "distractor_prompt",
    "partial_context_prompt",
    "negative_context_prompt",
]


def _parse_speech_qa(raw: str) -> tuple[str, str] | None:
    """Parse a 'Q: ...\\nA: ...' SpeechQA output. Returns None on SKIP or malformed."""
    if not raw:
        return None
    stripped = raw.strip()
    if stripped.upper() == "SKIP":
        return None
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    q_line = next((ln for ln in lines if ln.lower().startswith("q:")), None)
    a_line = next((ln for ln in lines if ln.lower().startswith("a:")), None)
    if not q_line or not a_line:
        return None
    question = q_line[2:].strip().lstrip(":").strip()
    answer = a_line[2:].strip().lstrip(":").strip()
    if not question or not answer:
        return None
    return question, answer


@dataclass
class InstructionPackerStage(ProcessingStage[AudioTask, AudioTask]):
    """Pack per-stage text outputs into a unified ``preference_instructions`` list.

    For each output field present on a task, emit one (or more, for
    context ASR) ``{"prompt": str, "target": str, "tags": {...}}`` entry
    into the ``preference_instructions`` field.  Each ``tags`` dict carries the
    ``type`` of variant that was picked and a ``target_lang`` (currently
    mirroring the per-sample source language).  Missing keys are silently
    skipped so the stage works regardless of which upstream stages ran.

    Args:
        output_key: Destination field name for the packed list.
        pnc_key / tn_key / itn_key / itn_no_disfluencies_key / captioning_key /
        code_switched_key / speech_qa_key: Source field names.  Use the
        same defaults as ``run_text_pipeline.py``.
        tn_key: TN-normalized (written→spoken) text.  Used as the target
            for the transcription pair (tagged ``tn``/``tn-pnc``) and for
            context-ASR prompt variants, in preference to ``pnc_text``;
            falls back to ``pnc_text`` when absent (e.g. TN stage disabled).
        context_asr_key: Source field name for the contextual ASR
            nested dict (produced by PR #14's ContextualASRPromptVariantStage).
            When present and the dict contains any of the six
            ``*_prompt`` fields, one entry per variant is emitted with
            the transcription target taken from ``transcription_target_key``.
        transcription_target_key: Field name whose value is used as the
            ``target`` for context-ASR prompt variants.  Defaults to
            ``pnc_text`` and is consulted after ``tn_key``; the resolution
            order is ``tn_key`` → ``transcription_target_key`` →
            ``cleaned_text``.
        source_lang_key: Field name holding the per-sample source
            language; copied into each entry's ``tags.target_lang``.
        notes_key: Field holding the ``additional_notes`` dict of
            per-stage decisions.
        pnc_note_key: Key within ``additional_notes`` recording the PnC
            stage decision.  The PnC entry is tagged ``type="tn-pnc"`` by
            default; it is downgraded to ``"tn"`` only when this note is
            present and does not start with ``"applied"`` (e.g.
            skipped/fallback/kept).
        sample_id_key: Stable per-sample identifier used to seed the RNG.
        seed: Base RNG seed combined with ``sample_id`` via SHA-256.
        skip_when_empty: If True (default), do not write the
            ``preference_instructions`` field when no pairs were produced.
    """

    name: str = "InstructionPacker"
    output_key: str = "preference_instructions"

    pnc_key: str = "pnc_text"
    tn_key: str = "tn_raw"
    itn_key: str = "itn_raw"
    itn_no_disfluencies_key: str = "itn_clean"
    captioning_key: str = "captioning_text"
    code_switched_key: str = "code_switched_text"
    speech_qa_key: str = "speech_qa_text"
    context_asr_key: str = "context_asr"
    transcription_target_key: str = "pnc_text"
    source_lang_key: str = "source_lang"

    notes_key: str = "additional_notes"
    pnc_note_key: str = "PnCRestoration"

    sample_id_key: str = "audio_filepath"
    seed: int = 42
    skip_when_empty: bool = True

    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 256

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        pass

    def teardown(self) -> None:
        pass

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_key]

    def _sample_seed(self, task: AudioTask) -> int:
        sample_id = str(task.data.get(self.sample_id_key, ""))
        h = hashlib.sha256(f"{self.seed}:{sample_id}".encode()).digest()
        return int.from_bytes(h[:8], "little")

    @staticmethod
    def _nonempty(text: Any) -> bool:  # noqa: ANN401
        return isinstance(text, str) and bool(text.strip())

    def _pnc_applied(self, data: dict[str, Any]) -> bool:
        """Assume PnC applied unless ``additional_notes[pnc_note_key]`` is present
        and does not start with ``"applied"`` (e.g. skipped/fallback/kept)."""
        notes = data.get(self.notes_key)
        if isinstance(notes, dict):
            note = notes.get(self.pnc_note_key)
            if isinstance(note, str) and note and not note.startswith("applied"):
                return False
        return True

    def _pack_one(self, task: AudioTask, rng: _random_module.Random) -> list[dict[str, Any]]:
        pairs: list[dict[str, Any]] = []
        data = task.data

        # target_lang currently mirrors the per-sample source language.
        target_lang = data.get(self.source_lang_key)

        def make_tags(instruction_type: str) -> dict[str, Any]:
            return {"type": instruction_type, "target_lang": target_lang}

        def add(prompts: list[str], target_key: str, instruction_type: str) -> None:
            target = data.get(target_key)
            if self._nonempty(target):
                pairs.append(
                    {"prompt": rng.choice(prompts), "target": target.strip(), "tags": make_tags(instruction_type)}
                )

        pnc_applied = self._pnc_applied(data)
        # Transcription target: prefer the TN-normalized (written→spoken) text
        # from the TN stage; fall back to pnc_text when TN didn't run / was skipped.
        transcription_key = self.tn_key if self._nonempty(data.get(self.tn_key)) else self.pnc_key
        add(_PNC_PROMPTS if pnc_applied else _TN_PROMPTS, transcription_key, "tn-pnc" if pnc_applied else "tn")
        add(_ITN_PROMPTS, self.itn_key, "itn")
        add(_ITN_NO_DISFL_PROMPTS, self.itn_no_disfluencies_key, "itn_no-disfluencies")
        add(_CAPTION_PROMPTS, self.captioning_key, "captioning")
        add(_CODE_SWITCH_PROMPTS, self.code_switched_key, "code_switched")

        qa = _parse_speech_qa(data.get(self.speech_qa_key, ""))
        if qa is not None:
            question, answer = qa
            pairs.append({"prompt": question, "target": answer, "tags": make_tags("speech_qa")})

        ctx = data.get(self.context_asr_key)
        if isinstance(ctx, dict):
            target = (
                data.get(self.tn_key) or data.get(self.transcription_target_key) or data.get("cleaned_text")
            )
            if self._nonempty(target):
                target = target.strip()
                for vkey in _CONTEXT_ASR_VARIANT_KEYS:
                    prompt = ctx.get(vkey)
                    if self._nonempty(prompt):
                        pairs.append(
                            {
                                "prompt": prompt.strip(),
                                "target": target,
                                "tags": make_tags(f"context_asr-{vkey.removesuffix('_prompt')}"),
                            }
                        )

        return pairs

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        n_emitted = 0
        for task in tasks:
            rng = _random_module.Random(self._sample_seed(task))  # noqa: S311
            pairs = self._pack_one(task, rng)
            if pairs:
                task.data[self.output_key] = pairs
                n_emitted += len(pairs)
            elif not self.skip_when_empty:
                task.data[self.output_key] = []

        logger.debug(f"{self.name}: packed {n_emitted} instruction pairs across {len(tasks)} tasks")
        return tasks
