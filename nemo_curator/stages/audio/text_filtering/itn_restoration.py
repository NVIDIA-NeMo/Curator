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

"""Inverse Text Normalization (ITN) stage using vLLM.

Converts spoken-form text (e.g. "fourteen dollars") to written form
(e.g. "$14") while preserving disfluencies and filler words.

Uses a text-only LLM (Qwen3.5 by default) with a system prompt that
specifies ITN conversion rules.  A validation step catches hallucinated
or unfaithful outputs and falls back to the original text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

_DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "itn_prompt.md"

_ALPHA_RE = re.compile(r"[a-zA-Z]+")
_ROMAN_RE = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
_VALID_ITN_ALPHA = frozenset({
    "st", "nd", "rd", "th",
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr",
    "ave", "blvd", "ln", "ct", "pl",
    "n", "s", "e", "w", "ne", "nw", "se", "sw",
    "kg", "km", "cm", "mm", "mg", "lb", "lbs", "oz",
    "ft", "mi", "ml", "mph", "h", "g", "m", "l",
    "am", "pm",
    "vs", "dept", "inc", "corp", "ltd", "co", "no",
})


def _validate_itn_output(input_text: str, output_text: str) -> tuple[bool, str]:
    """Check that ITN output is a faithful conversion, not a hallucination.

    Catches three failure modes:
      1. Word insertion  — output has more words than input
      2. Novel content   — output introduces alphabetic words not in input
      3. Excessive deletion — output dropped >50 % of words (summarisation)
    """
    in_words = input_text.lower().split()
    out_words = output_text.lower().split()

    if not in_words or not out_words:
        return True, "ok"

    if len(out_words) > len(in_words):
        return False, f"word_count_increase ({len(in_words)}->{len(out_words)})"

    in_alpha: set[str] = set()
    for w in in_words:
        for part in _ALPHA_RE.findall(w):
            in_alpha.add(part.lower())

    novel: list[str] = []
    for w in out_words:
        for part in _ALPHA_RE.findall(w):
            p = part.lower()
            if p in in_alpha or p in _VALID_ITN_ALPHA:
                continue
            if _ROMAN_RE.match(p):
                continue
            novel.append(p)

    if novel:
        return False, f"novel_words: {novel}"

    if len(in_words) >= 8 and len(out_words) < len(in_words) * 0.3:
        return False, f"excessive_deletion ({len(in_words)}->{len(out_words)})"

    return True, "ok"


@dataclass
class ITNRestorationStage(ProcessingStage[AudioTask, AudioTask]):
    """Inverse Text Normalization via batched LLM inference.

    Reads spoken-form text from ``text_key``, converts to written form
    using a vLLM-served text LLM, validates the output, and writes the
    result to ``output_text_key``.  Invalid outputs (hallucinations,
    insertions, excessive deletions) fall back to the original text and
    are flagged in ``itn_filtered_key``.

    The stage ships with a default ITN prompt bundled at
    ``prompts/itn_prompt.md``.  Override with ``prompt_text`` (inline
    string) or ``prompt_file`` (path to a markdown file).

    Uses ``process_batch`` for efficient batched GPU inference via vLLM
    with prefix caching enabled (the shared system prompt is cached
    across all requests).

    Args:
        model_id: HuggingFace model identifier for the text LLM.
        prompt_text: System prompt string.  Takes precedence over
            ``prompt_file``.
        prompt_file: Path to a file containing the system prompt.
            Falls back to the bundled default if neither ``prompt_text``
            nor ``prompt_file`` is set.
        text_key: Input manifest key to read spoken-form text from.
        output_text_key: Output manifest key for the ITN result.
        skip_me_key: Key used to flag entries for downstream filtering.
        itn_filtered_key: Key where validation failure reasons are stored.
        enable_validation: Run output validation and fall back on failure.
        tensor_parallel_size: GPUs for tensor parallelism (``None`` =
            auto-detect).
        max_output_tokens: Maximum tokens to generate per sample.
        max_model_len: Maximum context length passed to vLLM.
        max_num_seqs: Maximum concurrent sequences in vLLM.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        kv_cache_dtype: KV-cache dtype (``fp8`` halves memory, 2x
            concurrent sequences on Hopper).
    """

    name: str = "ITNRestoration"
    model_id: str = "Qwen/Qwen3.5-35B-A3B-FP8"
    prompt_text: str | None = None
    prompt_file: str | None = None
    text_key: str = "pnc_text"
    output_text_key: str = "itn_text"
    skip_me_key: str = "_skip_me"
    itn_filtered_key: str = "itn_filtered"
    enable_validation: bool = True
    tensor_parallel_size: int | None = None
    max_output_tokens: int = 4096
    max_model_len: int = 32768
    max_num_seqs: int = 256
    gpu_memory_utilization: float = 0.95
    kv_cache_dtype: str = "fp8"
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 64

    _llm: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _sampling_params: Any = field(default=None, init=False, repr=False)
    _system_prompt: str = field(default="", init=False, repr=False)
    _n_processed: int = field(default=0, init=False, repr=False)
    _n_filtered: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        tp = self.tensor_parallel_size
        if tp and tp > 0:
            self.resources = Resources(gpus=float(tp))

    # ------------------------------------------------------------------
    # Prompt resolution
    # ------------------------------------------------------------------

    def _resolve_prompt(self) -> str:
        if self.prompt_text:
            return self.prompt_text

        path = Path(self.prompt_file) if self.prompt_file else _DEFAULT_PROMPT_PATH
        if not path.exists():
            msg = f"ITN prompt file not found: {path}"
            raise FileNotFoundError(msg)
        return path.read_text(encoding="utf-8").strip()

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------

    def _init_model(self) -> None:
        if not VLLM_AVAILABLE:
            msg = "vLLM is required for ITNRestorationStage. pip install vllm"
            raise ImportError(msg)

        self._system_prompt = self._resolve_prompt()

        from nemo_curator.utils.gpu_utils import get_gpu_count

        tp = self.tensor_parallel_size if self.tensor_parallel_size else get_gpu_count()

        logger.info(
            "ITN: loading %s (tp=%d, max_model_len=%d, kv_cache_dtype=%s)",
            self.model_id, tp, self.max_model_len, self.kv_cache_dtype,
        )

        self._llm = LLM(
            model=self.model_id,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=tp,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=8192,
            enable_prefix_caching=True,
            prefix_caching_hash_algo="xxhash",
            kv_cache_dtype=self.kv_cache_dtype,
            enforce_eager=False,
            seed=1234,
        )
        self._tokenizer = self._llm.get_tokenizer()
        self._sampling_params = SamplingParams(
            temperature=0,
            max_tokens=self.max_output_tokens,
        )

        logger.info(
            "ITN: model ready (prefix_caching=True, prompt=%d chars)",
            len(self._system_prompt),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        self._init_model()

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._llm is None:
            self._init_model()

    def teardown(self) -> None:
        if self._n_processed:
            logger.info(
                "ITN: processed %d, filtered %d (%.1f%%)",
                self._n_processed,
                self._n_filtered,
                100.0 * self._n_filtered / self._n_processed if self._n_processed else 0,
            )
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._tokenizer = None
            self._sampling_params = None

    # ------------------------------------------------------------------
    # I/O contract
    # ------------------------------------------------------------------

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.skip_me_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_text_key, self.itn_filtered_key]

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def _format_prompt(self, user_text: str) -> str:
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_text},
        ]
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if not tasks:
            return []

        if self._llm is None:
            msg = "Model not initialised — setup() was not called"
            raise RuntimeError(msg)

        valid_indices: list[int] = []
        prompts: list[str] = []

        for i, task in enumerate(tasks):
            text = task.data.get(self.text_key, "")
            skip = task.data.get(self.skip_me_key, "")
            if not text or not text.strip() or skip:
                task.data[self.output_text_key] = text
                continue
            valid_indices.append(i)
            prompts.append(self._format_prompt(text))

        if prompts:
            outputs = self._llm.generate(
                prompts,
                sampling_params=self._sampling_params,
                use_tqdm=False,
            )

            for seq_idx, task_idx in enumerate(valid_indices):
                task = tasks[task_idx]
                input_text = task.data[self.text_key]
                itn_text = outputs[seq_idx].outputs[0].text.strip()

                if self.enable_validation:
                    ok, reason = _validate_itn_output(input_text, itn_text)
                    if ok:
                        task.data[self.output_text_key] = itn_text
                    else:
                        task.data[self.output_text_key] = input_text
                        task.data[self.itn_filtered_key] = reason
                        self._n_filtered += 1
                else:
                    task.data[self.output_text_key] = itn_text

                self._n_processed += 1

        logger.debug("ITN: batch of %d tasks (%d inferred)", len(tasks), len(prompts))
        return tasks
