# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Generic text-to-text LLM stage with per-process model caching.

Each pipeline stage runs as its own actor (its own process) under the
Ray Data / Xenna / Ray actor-pool executors, so every stage loads and
owns its own vLLM engine and may use its own ``max_model_len`` /
sampling config. The module-level ``_model_cache`` only deduplicates
loads *within a single process* (keyed by ``model_id``); it does not
share one engine across stages.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

from nemo_curator.stages.audio.pipeline_utils import set_note
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# ── Per-process model cache ─────────────────────────────────────────
# Module-level cache keyed by model_id. Dedupes loads WITHIN one actor
# process only — each stage is its own actor/process, so this never
# shares an engine across stages (each stage keeps its own max_model_len).

_model_cache: dict[str, Any] = {}
_cache_lock = threading.Lock()


def _isolate_compile_caches() -> None:
    """Give this actor process its own torch.compile / inductor / triton caches.

    Multiple vLLM engines co-located on one node otherwise race on the *shared*
    default cache dirs (``~/.cache``, ``/tmp/torchinductor_<user>``,
    ``/tmp/triton``). That race corrupts the inductor cache pickle
    ("pickle data was truncated" / "CompiledFxGraph has no compiled_fn_runner"),
    kills the EngineCore, and its restart then hangs at CUDA-graph capture —
    wedging the whole pipeline (idle GPUs → killed by the cluster idle reaper).
    Keying every cache dir by PID isolates each engine. Each Ray actor runs one
    engine in its own process, and vLLM's spawned EngineCore subprocess inherits
    these env vars, so all of an actor's (re)starts share one private cache while
    different actors never collide. Must run before the ``LLM(...)`` constructor.
    """
    root = os.environ.get("COMPILE_CACHE_ROOT", os.environ.get("TMPDIR", "/tmp"))
    base = os.path.join(root, f"compile_cache_{os.getpid()}")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(base, "inductor")
    os.environ["TRITON_CACHE_DIR"] = os.path.join(base, "triton")
    os.environ["VLLM_CACHE_ROOT"] = os.path.join(base, "vllm")
    try:
        os.makedirs(base, exist_ok=True)
    except OSError as exc:  # non-fatal — torch/triton/vllm create them lazily too
        logger.warning(f"_isolate_compile_caches: could not pre-create {base}: {exc}")
    logger.info(f"Per-process compile caches under {base} (pid={os.getpid()})")


def _get_or_load_model(  # noqa: PLR0913
    model_id: str,
    tensor_parallel_size: int,
    max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    kv_cache_dtype: str,
) -> tuple[Any, Any]:
    """Return ``(llm, tokenizer)`` from cache or load a new engine."""
    with _cache_lock:
        if model_id in _model_cache:
            logger.info(f"TextLLMStage: reusing cached model {model_id}")
            entry = _model_cache[model_id]
            return entry["llm"], entry["tokenizer"]

        if not VLLM_AVAILABLE:
            msg = "vLLM is required for TextLLMStage. pip install vllm"
            raise ImportError(msg)

        # Isolate this process's compile caches BEFORE building the engine, so
        # co-located stages don't race on the shared inductor/triton cache.
        _isolate_compile_caches()

        # enforce_eager skips torch.compile + CUDA-graph capture. Set
        # VLLM_ENFORCE_EAGER=1 to avoid the graph-capture cost on every engine
        # (re)spawn and the CUDA-graph-replay hang class that wedges the pipeline
        # when Ray Data tears down / respawns an actor mid-stream.
        enforce_eager = os.environ.get("VLLM_ENFORCE_EAGER", "0").lower() in ("1", "true", "yes")

        # Optionally force-disable vLLM V1 async scheduling — the
        # `step_with_batch_queue` path that deadlocks (SM-100%/mem-0% spin) after
        # Ray Data tears down/respawns an actor mid-stream. VLLM_ASYNC_SCHEDULING=0
        # forces synchronous stepping; unset → vLLM default (auto-on in V1).
        _async = os.environ.get("VLLM_ASYNC_SCHEDULING", "").strip().lower()
        async_kwargs: dict[str, Any] = {}
        if _async in ("0", "false", "no"):
            async_kwargs["async_scheduling"] = False
        elif _async in ("1", "true", "yes"):
            async_kwargs["async_scheduling"] = True

        logger.info(
            "TextLLMStage: loading %s (tp=%d, max_model_len=%d, kv_cache=%s, enforce_eager=%s)",
            model_id,
            tensor_parallel_size,
            max_model_len,
            kv_cache_dtype,
            enforce_eager,
        )
        llm = LLM(
            model=model_id,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=8192,
            enable_prefix_caching=True,
            prefix_caching_hash_algo="xxhash",
            kv_cache_dtype=kv_cache_dtype,
            enforce_eager=enforce_eager,
            seed=1234,
            **async_kwargs,
        )
        tokenizer = llm.get_tokenizer()
        _model_cache[model_id] = {"llm": llm, "tokenizer": tokenizer}
        logger.info("TextLLMStage: model %s ready", model_id)
        return llm, tokenizer


@dataclass
class TextLLMStage(ProcessingStage[AudioTask, AudioTask]):
    """Generic prompt-based text-to-text stage via vLLM.

    Reads text from ``text_key``, sends it through a vLLM-served LLM
    with the configured system prompt, and writes the result to
    ``output_text_key``.  Optionally validates the output and falls
    back to the original text.

    Each stage runs as its own actor/process and loads its own vLLM
    engine (the module-level cache only dedupes within one process), so
    each stage may use its own ``max_model_len`` / sampling config.

    Args:
        model_id: HuggingFace model identifier.
        prompt_text: System prompt string (takes precedence over prompt_file).
        prompt_file: Path to a file containing the system prompt.
        text_key: Input field to read text from.
        output_text_key: Output field to write the result to.
        skip_me_key: Field that flags entries to skip.
        notes_key: Field for additional_notes tracking.
        enable_validation: Validate output (word count checks).
        max_deletion_ratio: Max fraction of words that can be removed.
        tensor_parallel_size: GPUs for TP (None = auto-detect).
        max_output_tokens: Max tokens to generate per sample.
        max_model_len: Max context length for vLLM.
        max_num_seqs: Max concurrent sequences in vLLM.
        gpu_memory_utilization: Fraction of GPU memory for vLLM.
        kv_cache_dtype: KV-cache dtype (fp8 for Hopper).
        num_workers_override: Explicit worker count for Xenna.
    """

    name: str = "TextLLM"
    model_id: str = "Qwen/Qwen3.5-35B-A3B-FP8"
    prompt_text: str | None = None
    prompt_file: str | None = None
    text_key: str = "pnc_text"
    output_text_key: str = "output_text"
    skip_me_key: str = "_skipme"
    notes_key: str = "additional_notes"
    enable_validation: bool = True
    skip_if_output_exists: bool = False
    max_deletion_ratio: float = 0.3
    min_words_for_deletion_check: int = 8
    tensor_parallel_size: int | None = None
    max_output_tokens: int = 512
    max_model_len: int = 2048
    max_num_seqs: int = 64
    gpu_memory_utilization: float = 0.95
    kv_cache_dtype: str = "fp8"
    num_workers_override: int | None = None
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

    def num_workers(self) -> int | None:
        return self.num_workers_override

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.num_workers_override is not None:
            spec["num_workers"] = self.num_workers_override
        return spec

    # ── Prompt resolution ────────────────────────────────────────────

    def _resolve_prompt(self) -> str:
        if self.prompt_text:
            return self.prompt_text
        if self.prompt_file:
            path = Path(self.prompt_file)
            if not path.exists():
                msg = f"Prompt file not found: {path}"
                raise FileNotFoundError(msg)
            return path.read_text(encoding="utf-8").strip()
        msg = "Either prompt_text or prompt_file must be provided"
        raise ValueError(msg)

    # ── Model lifecycle ──────────────────────────────────────────────

    def _init_model(self) -> None:
        self._system_prompt = self._resolve_prompt()

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
        self._sampling_params = SamplingParams(temperature=0, max_tokens=self.max_output_tokens)
        logger.info(
            "%s: ready (prompt=%d chars, output_key=%s)", self.name, len(self._system_prompt), self.output_text_key
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
                "%s: processed %d, filtered %d (%.1f%%)",
                self.name,
                self._n_processed,
                self._n_filtered,
                100.0 * self._n_filtered / self._n_processed if self._n_processed else 0,
            )

    # ── I/O contract ─────────────────────────────────────────────────

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.skip_me_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_text_key]

    # ── Prompt formatting ────────────────────────────────────────────

    def _format_prompt(self, user_text: str, task_data: dict | None = None) -> str:
        prompt_template = self._system_prompt

        if "{language}" in prompt_template:
            lang = task_data.get("source_lang", "English") if task_data else "English"
            prompt_template = prompt_template.replace("{language}", lang)

        if "{text}" in prompt_template:
            prompt_template = prompt_template.replace("{text}", user_text)
            messages = [{"role": "user", "content": prompt_template}]
        else:
            messages = [
                {"role": "system", "content": prompt_template},
                {"role": "user", "content": user_text},
            ]

        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    # ── Validation ───────────────────────────────────────────────────

    _ALPHA_RE = re.compile(r"[a-zA-Z]+")
    _ROMAN_RE = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
    _VALID_ITN_ALPHA = frozenset(
        {
            "st",
            "nd",
            "rd",
            "th",
            "dr",
            "mr",
            "mrs",
            "ms",
            "prof",
            "sr",
            "jr",
            "ave",
            "blvd",
            "ln",
            "ct",
            "pl",
            "n",
            "s",
            "e",
            "w",
            "ne",
            "nw",
            "se",
            "sw",
            "kg",
            "km",
            "cm",
            "mm",
            "mg",
            "lb",
            "lbs",
            "oz",
            "ft",
            "mi",
            "ml",
            "mph",
            "h",
            "g",
            "m",
            "l",
            "am",
            "pm",
            "vs",
            "dept",
            "inc",
            "corp",
            "ltd",
            "co",
            "no",
        }
    )

    def _check_novel_words(self, in_words: list[str], out_words: list[str]) -> list[str]:
        in_alpha: set[str] = set()
        for w in in_words:
            for part in self._ALPHA_RE.findall(w):
                in_alpha.add(part.lower())
        novel: list[str] = []
        for w in out_words:
            for part in self._ALPHA_RE.findall(w):
                p = part.lower()
                if p in in_alpha or p in self._VALID_ITN_ALPHA:
                    continue
                if self._ROMAN_RE.match(p):
                    continue
                novel.append(p)
        return novel

    def _validate(self, input_text: str, output_text: str) -> tuple[bool, str]:
        in_words = input_text.lower().split()
        out_words = output_text.lower().split()
        if not in_words or not out_words:
            return True, "ok"
        if len(out_words) > len(in_words):
            return False, f"word_count_increase ({len(in_words)}->{len(out_words)})"
        novel = self._check_novel_words(in_words, out_words)
        if novel:
            return False, f"novel_words: {novel}"
        if (
            len(in_words) >= self.min_words_for_deletion_check
            and len(out_words) < len(in_words) * self.max_deletion_ratio
        ):
            return False, f"excessive_deletion ({len(in_words)}->{len(out_words)})"
        return True, "ok"

    # ── Processing ───────────────────────────────────────────────────

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
            if self.skip_if_output_exists and task.data.get(self.output_text_key):
                set_note(task.data, self.name, "skipped (output exists)", self.notes_key)
                continue
            if skip:
                task.data[self.output_text_key] = ""
                set_note(task.data, self.name, "skipped (flagged)", self.notes_key)
                continue
            if not text or not text.strip():
                task.data[self.output_text_key] = text
                set_note(task.data, self.name, "skipped (empty)", self.notes_key)
                continue
            valid_indices.append(i)
            prompts.append(self._format_prompt(text, task.data))

        if prompts:
            outputs = self._llm.generate(prompts, sampling_params=self._sampling_params, use_tqdm=False)

            for seq_idx, task_idx in enumerate(valid_indices):
                task = tasks[task_idx]
                input_text = task.data[self.text_key]
                result_text = outputs[seq_idx].outputs[0].text.strip()

                if self.enable_validation:
                    ok, reason = self._validate(input_text, result_text)
                    if ok:
                        task.data[self.output_text_key] = result_text
                        note = "applied (modified)" if result_text != input_text else "applied (unchanged)"
                    else:
                        task.data[self.output_text_key] = input_text
                        self._n_filtered += 1
                        note = f"fallback ({reason})"
                else:
                    task.data[self.output_text_key] = result_text
                    note = "applied (modified)" if result_text != input_text else "applied (unchanged)"

                set_note(task.data, self.name, note, self.notes_key)
                self._n_processed += 1

        logger.debug("%s: batch of %d tasks (%d inferred)", self.name, len(tasks), len(prompts))
        return tasks
