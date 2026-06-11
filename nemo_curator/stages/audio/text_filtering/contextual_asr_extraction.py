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

"""Contextual ASR entity extraction stage using vLLM.

Reads a transcript from ``text_key`` (default: ``pnc_text``) and uses a
text-only LLM to extract contextual-ASR biasing information: domain
labels, named entities across 9 category buckets, distractor terms,
confidence scores, speaking style, and difficulty estimation.

The user prompt template supports both ``{transcript}`` and
``{source_lang}`` placeholders.  The per-sample source language is read
from ``source_lang_key`` (default ``source_lang``) so the extraction LLM
knows which language the transcript is in and preserves entity names in
their original script.

The extracted fields are written as a nested dict under ``output_key``
(default: ``context_asr``).  On JSON parse failure the key is set to
``None`` and a ``json_parse_failed`` note is written via ``set_note``.

A downstream :class:`ContextualASRPromptVariantStage` (CPU-only) reads
the extraction output and appends six prompt-variant fields to the same
nested dict.

Engine ownership
----------------
Each pipeline stage runs as its own actor (its own process) under the
Ray Data / Xenna / Ray actor-pool executors, so this stage loads and
owns its own vLLM engine via ``_get_or_load_model`` — independent of the
other text-LLM stages. It therefore uses its OWN engine config
(``max_model_len``, ``max_num_seqs``, etc.) and runs with a larger
context window than the lightweight stages: see
``--context_asr_max_model_len`` in :file:`run_text_pipeline.py` (default
8192) vs the global ``--max_model_len`` (2048).

The module-level ``_model_cache`` (keyed by the bare ``model_id``) only
deduplicates loads *within a single process*; it does not share one
engine across stages. The "first caller wins" behaviour would only
matter if several stages were co-located in one process, which does not
happen with the per-stage-actor executors used here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

from nemo_curator.stages.audio.pipeline_utils import set_note
from nemo_curator.stages.audio.text_filtering.text_llm_stage import _get_or_load_model
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

try:
    from vllm import SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

_DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "contextual_asr_prompt.md"

_ENTITY_CATEGORY_KEYS = [
    "person_name",
    "company_name",
    "product_name",
    "drug_name",
    "location_name",
    "organization_name",
    "event_name",
    "technical_term",
    "abbreviation",
]

_VALID_SPEAKING_STYLES = frozenset(
    {
        "formal",
        "conversational",
        "technical",
        "narrative",
        "instructional",
    }
)

_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _parse_json_response(raw: str) -> dict | None:
    sanitized = raw.strip()
    sanitized = _THINK_BLOCK_RE.sub("", sanitized).strip()
    if sanitized.startswith("```"):
        sanitized = sanitized.split("\n", 1)[1] if "\n" in sanitized else sanitized[3:]
    if sanitized.endswith("```"):
        sanitized = sanitized.rsplit("```", 1)[0]
    sanitized = sanitized.strip()
    sanitized = _BLOCK_COMMENT_RE.sub("", sanitized)
    sanitized = _LINE_COMMENT_RE.sub("", sanitized)
    sanitized = _TRAILING_COMMA_RE.sub(r"\1", sanitized)
    try:
        return json.loads(sanitized, strict=False)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(sanitized)
        if match:
            try:
                return json.loads(match.group(), strict=False)
            except json.JSONDecodeError:
                return None
    return None


def _coerce_int_in_range(value: Any, lo: int, hi: int, default: int) -> int:  # noqa: ANN401
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, iv))


def _coerce_str_list(value: Any) -> list[str]:  # noqa: ANN401
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if isinstance(v, (str, int, float)) and str(v).strip()]


def _normalize_extraction(extracted: dict | None) -> dict:
    if not isinstance(extracted, dict):
        extracted = {}

    coarse_terms = _coerce_str_list(extracted.get("coarse_context_terms"))[:3]
    fine_terms = _coerce_str_list(extracted.get("fine_context_terms"))
    distractor_terms = _coerce_str_list(extracted.get("distractor_terms"))[:8]

    raw_cats = extracted.get("entity_categories") or {}
    if not isinstance(raw_cats, dict):
        raw_cats = {}
    entity_categories = {k: _coerce_str_list(raw_cats.get(k)) for k in _ENTITY_CATEGORY_KEYS}

    cats_union = {t for terms in entity_categories.values() for t in terms}
    if cats_union and not fine_terms:
        fine_terms = sorted(cats_union)

    confidence_coarse = _coerce_int_in_range(extracted.get("confidence_coarse"), 1, 5, 1)
    confidence_fine = _coerce_int_in_range(extracted.get("confidence_fine"), 1, 5, 1)
    estimated_difficulty = _coerce_int_in_range(extracted.get("estimated_difficulty"), 1, 5, 3)

    speaking_style = str(extracted.get("speaking_style", "conversational")).strip().lower()
    if speaking_style not in _VALID_SPEAKING_STYLES:
        speaking_style = "conversational"

    return {
        "coarse_context_terms": coarse_terms,
        "fine_context_terms": fine_terms,
        "entity_categories": entity_categories,
        "distractor_terms": distractor_terms,
        "confidence_coarse": confidence_coarse,
        "confidence_fine": confidence_fine,
        "speaking_style": speaking_style,
        "estimated_difficulty": estimated_difficulty,
    }


def _load_prompt_sections(prompt_path: Path) -> tuple[str, str]:
    raw = prompt_path.read_text(encoding="utf-8")
    if "# SYSTEM_PROMPT" not in raw or "# USER_PROMPT_TEMPLATE" not in raw:
        msg = f"{prompt_path} must contain both '# SYSTEM_PROMPT' and '# USER_PROMPT_TEMPLATE' headers."
        raise ValueError(msg)
    _, after_sys = raw.split("# SYSTEM_PROMPT", 1)
    sys_body, user_body = after_sys.split("# USER_PROMPT_TEMPLATE", 1)
    return sys_body.strip(), user_body.strip()


@dataclass
class ContextualASRExtractionStage(ProcessingStage[AudioTask, AudioTask]):
    """Extract contextual-ASR biasing information via batched LLM inference.

    Reads transcript from ``text_key``, calls a text LLM to extract
    domain labels, named entities, distractors, and metadata.  Results
    are written as a nested dict under ``output_key``.

    On JSON parse failure, ``output_key`` is set to ``None`` and the
    failure is recorded via ``set_note(task.data, name, "json_parse_failed")``.

    Runs as its own actor/process and loads its own vLLM engine via the
    per-process ``_model_cache`` in :mod:`text_llm_stage`; it does not
    share an engine with the other stages, so it uses its own
    ``max_model_len`` (typically larger — see ``--context_asr_max_model_len``).

    Args:
        model_id: HuggingFace model identifier for the text LLM.
        prompt_file: Path to the system+user prompt markdown file.
            Falls back to the bundled default if not set.
        text_key: Input manifest key holding the transcript.  The
            transcript value is bound to the ``{transcript}`` placeholder
            in the user prompt template.
        source_lang_key: Manifest key holding the source language (display
            name or code) used to fill the ``{source_lang}`` placeholder
            in the user prompt template.  Templates that don't reference
            the placeholder still work — the value is simply unused.
        default_source_lang: Fallback string used when ``source_lang_key``
            is missing or empty on a sample.
        output_key: Output manifest key for the nested extraction dict.
        skip_me_key: Key used to check whether entry is flagged.
        notes_key: Key holding the ``additional_notes`` dict that
            :func:`set_note` writes into.
        tensor_parallel_size: GPUs for tensor parallelism (None = auto).
        max_output_tokens: Maximum tokens to generate per sample.
        max_model_len: Maximum context length for this stage's own vLLM
            engine. Independent of the other stages' engines, so it can be
            larger (default 8192) than the global ``--max_model_len``.
        max_num_seqs: Maximum concurrent sequences in this stage's engine.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        kv_cache_dtype: KV-cache dtype for vLLM.
        num_workers_override: Explicit worker count for Xenna.
        batch_size: Number of samples per inference batch.
    """

    name: str = "ContextualASRExtraction"
    model_id: str = "Qwen/Qwen3.5-35B-A3B-FP8"
    prompt_file: str | None = None
    text_key: str = "pnc_text"
    source_lang_key: str = "source_lang"
    default_source_lang: str = "English"
    output_key: str = "context_asr"
    skip_me_key: str = "_skipme"
    notes_key: str = "additional_notes"
    tensor_parallel_size: int | None = None
    max_output_tokens: int = 2048
    max_model_len: int = 8192
    max_num_seqs: int = 16
    gpu_memory_utilization: float = 0.95
    kv_cache_dtype: str = "fp8"
    num_workers_override: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 64

    _llm: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _sampling_params: Any = field(default=None, init=False, repr=False)
    _system_prompt: str = field(default="", init=False, repr=False)
    _user_prompt_template: str = field(default="", init=False, repr=False)
    _n_processed: int = field(default=0, init=False, repr=False)
    _n_failed: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        tp = self.tensor_parallel_size
        if tp and tp > 0:
            self.resources = Resources(gpus=float(tp))

    def num_workers(self) -> int | None:
        return self.num_workers_override

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.num_workers_override is not None:
            spec["num_workers"] = self.num_workers_override
        return spec

    def _resolve_prompts(self) -> tuple[str, str]:
        path = Path(self.prompt_file) if self.prompt_file else _DEFAULT_PROMPT_PATH
        if not path.exists():
            msg = f"Context ASR prompt file not found: {path}"
            raise FileNotFoundError(msg)
        return _load_prompt_sections(path)

    def _init_model(self) -> None:
        if not VLLM_AVAILABLE:
            msg = "vLLM is required for ContextualASRExtractionStage. pip install vllm"
            raise ImportError(msg)

        self._system_prompt, self._user_prompt_template = self._resolve_prompts()

        from nemo_curator.utils.gpu_utils import get_gpu_count

        tp = self.tensor_parallel_size or get_gpu_count()

        self._llm, self._tokenizer = _get_or_load_model(
            model_id=self.model_id,
            tensor_parallel_size=tp,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            gpu_memory_utilization=self.gpu_memory_utilization,
            kv_cache_dtype=self.kv_cache_dtype,
        )
        self._sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=self.max_output_tokens,
        )

        logger.info(
            "%s: ready (system_prompt=%d chars, output_key=%s)",
            self.name,
            len(self._system_prompt),
            self.output_key,
        )

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        pass

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._llm is None:
            self._init_model()

    def teardown(self) -> None:
        if self._n_processed:
            logger.info(
                "%s: processed %d, failed %d (%.1f%%)",
                self.name,
                self._n_processed,
                self._n_failed,
                100.0 * self._n_failed / self._n_processed if self._n_processed else 0,
            )

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.source_lang_key, self.skip_me_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_key]

    def _format_prompt(self, transcript: str, source_lang: str) -> str:
        # The user template may reference {transcript}, {source_lang}, or
        # both.  Only pass placeholders that actually appear so older
        # prompt files (pre-source_lang) keep working unchanged.
        fmt: dict[str, str] = {}
        if "{transcript}" in self._user_prompt_template:
            fmt["transcript"] = transcript
        if "{source_lang}" in self._user_prompt_template:
            fmt["source_lang"] = source_lang
        user_content = self._user_prompt_template.format(**fmt) if fmt else self._user_prompt_template
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []

        if self._llm is None:
            msg = "Model not initialised — setup() was not called"
            raise RuntimeError(msg)

        valid_indices: list[int] = []
        prompts: list[str] = []

        for i, task in enumerate(tasks):
            text = task.data.get(self.text_key, "")
            skip = task.data.get(self.skip_me_key, "")
            if skip:
                task.data[self.output_key] = None
                set_note(task.data, self.name, "skipped (flagged)", self.notes_key)
                continue
            if not text or not text.strip():
                task.data[self.output_key] = None
                set_note(task.data, self.name, "skipped (empty)", self.notes_key)
                continue
            source_lang = task.data.get(self.source_lang_key) or self.default_source_lang
            valid_indices.append(i)
            prompts.append(self._format_prompt(text.strip(), str(source_lang)))

        if prompts:
            outputs = self._llm.generate(
                prompts,
                sampling_params=self._sampling_params,
                use_tqdm=False,
            )

            for seq_idx, task_idx in enumerate(valid_indices):
                task = tasks[task_idx]
                raw = outputs[seq_idx].outputs[0].text.strip()
                parsed = _parse_json_response(raw)

                self._n_processed += 1
                if parsed is None:
                    task.data[self.output_key] = None
                    set_note(task.data, self.name, "json_parse_failed", self.notes_key)
                    self._n_failed += 1
                else:
                    task.data[self.output_key] = _normalize_extraction(parsed)
                    set_note(task.data, self.name, "extracted", self.notes_key)

        logger.debug(
            "%s: batch of %d tasks (%d inferred, %d failed total)",
            self.name,
            len(tasks),
            len(prompts),
            self._n_failed,
        )
        return tasks
