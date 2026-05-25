# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Punctuation and Capitalization (PNC) stages.

Contains two PNC stages:

* :class:`PNCwithvLLMInferenceStage` — vLLM-backed LLM PNC.
* :class:`CleanLLMOutputStage` — post-processing for vLLM PNC output.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger
from nemo.collections.asr.metrics.wer import word_error_rate_detail

from nemo_curator.models.vllm_model import VLLMInference
from nemo_curator.stages.audio.tagging.utils import load_vocab_file
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata


@dataclass
class PNCwithvLLMInferenceStage(ProcessingStage[AudioTask, AudioTask]):
    """Punctuation and capitalisation using a vLLM language model.

    Every entry is treated as an audio entry.  When a ``segments`` list is
    present the stage processes each segment's text individually; otherwise it
    processes the top-level ``text_key``.

    The generated (punctuated / capitalised) text is stored under
    ``generation_field`` in each segment or entry dict.

    Args:
        text_key:            Manifest key holding the raw text.
        segments_key:        Manifest key for the segments list.
        prompt:              Static YAML-style chat prompt (dict of role→content).
        prompt_field:        Per-entry key that holds the prompt template.
        prompt_file:         Path to a YAML file with the prompt structure.
        generation_field:    Output key for the generated text.
        model_params:        Kwargs forwarded to ``vllm.LLM()``.
        sampling_params:     Kwargs forwarded to ``vllm.SamplingParams()``.
        chat_template_params: Kwargs forwarded to ``tokenizer.apply_chat_template()``.
        use_chat_api:        When *True*, use ``llm.chat()`` instead of
                             ``llm.generate()``.  Defaults to *False*.
        inference_batch_size: Max prompts per ``llm.generate()`` call.
    """

    text_key: str = "text"
    segments_key: str = "segments"

    prompt: dict[str, str] | None = None
    prompt_field: str | None = None
    prompt_file: str | None = None
    generation_field: str = "generation"

    model_params: dict[str, Any] = field(default_factory=dict)
    sampling_params: dict[str, Any] = field(default_factory=dict)
    chat_template_params: dict[str, Any] = field(default_factory=dict)
    use_chat_api: bool = False

    inference_batch_size: int = 10000
    batch_size: int = 64

    name: str = "PNCwithvLLMInference"
    resources: Resources = field(default_factory=lambda: Resources(gpus=1))

    _vllm: VLLMInference | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._vllm = VLLMInference(
            prompt=self.prompt,
            prompt_field=self.prompt_field,
            prompt_file=self.prompt_file,
            generation_field=self.generation_field,
            model=dict(self.model_params),
            inference=dict(self.sampling_params),
            apply_chat_template=dict(self.chat_template_params),
            use_chat_api=self.use_chat_api,
            device="cuda" if self.resources.requires_gpu else "cpu",
        )

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        """Pre-download model weights and tokenizer (called once per node)."""
        self._vllm.setup_on_node()

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Load tokenizer and vLLM engine (called once per worker)."""
        self._vllm.setup()

    def teardown(self) -> None:
        if self._vllm is not None:
            self._vllm.clean_up()
            self._vllm = None

    def _collect_prompts(self, data: dict) -> list[tuple[str | list, list]]:
        """Collect ``(prompt, key_path)`` pairs from a manifest entry.

        If ``segments`` are present, each segment with non-empty text produces
        one prompt (key_path ``[segments_key, idx]``).  Otherwise a single
        prompt is produced from the top-level text (key_path ``[]``).
        """
        items: list[tuple[str | list, list]] = []
        if self.segments_key in data:
            for idx, segment in enumerate(data[self.segments_key]):
                if self.text_key in segment and segment[self.text_key] != "":
                    prompt = self._vllm.get_entry_prompt(segment)
                    items.append((prompt, [self.segments_key, idx]))
        elif self.text_key in data and data[self.text_key] != "":
            prompt = self._vllm.get_entry_prompt(data)
            items.append((prompt, []))
        return items

    @staticmethod
    def _store_generation(data: dict, key_path: list, text: str, gen_field: str) -> None:
        """Write *text* into *data* following *key_path*."""
        target = data
        for key in key_path:
            target = target[key]
        target[gen_field] = text

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.segments_key, self.text_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.generation_field]

    def process(self, task: AudioTask) -> AudioTask:
        msg = f"{self.name} is a GPU/batched inference stage. Use process_batch() instead."
        raise NotImplementedError(msg)

    @staticmethod
    def _extract_generated_text(output: Any, prompt_idx: int, key_path: list) -> str | None:  # noqa: ANN401
        """Safely extract generated text from a vLLM RequestOutput.

        Returns ``None`` if the output has no completions, logging a warning
        with enough context to identify the failing prompt.
        """
        if not output.outputs:
            logger.warning(
                f"[PNCwithvLLMInference] Empty completion at prompt_idx={prompt_idx}, key_path={key_path}; skipping."
            )
            return None
        return output.outputs[0].text

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Batch-process multiple AudioTasks with a single vLLM call for efficiency."""
        t0 = time.perf_counter()

        all_prompts: list = []
        result_map: list[tuple[int, list]] = []

        for task_idx, task in enumerate(tasks):
            for prompt, key_path in self._collect_prompts(task.data):
                all_prompts.append(prompt)
                result_map.append((task_idx, key_path))

        if not all_prompts:
            return tasks

        outputs: list = []
        for i in range(0, len(all_prompts), self.inference_batch_size):
            batch = all_prompts[i : i + self.inference_batch_size]
            outputs.extend(self._vllm.process_batch(batch))

        for idx, ((task_idx, key_path), output) in enumerate(zip(result_map, outputs, strict=True)):
            text = self._extract_generated_text(output, idx, key_path)
            if text is not None:
                self._store_generation(tasks[task_idx].data, key_path, text, self.generation_field)

        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "entries_processed": len(tasks),
                "prompts_generated": len(all_prompts),
            }
        )

        return tasks


@dataclass
class CleanLLMOutputStage(ProcessingStage[AudioTask, AudioTask]):
    """Clean vLLM PNC output and decide whether to fall back to BERT PNC.

    Compares the LLM-generated punctuated text against the original ASR
    prediction using CER.  When the CER exceeds the threshold, the output
    contains digits, or the text has invalid characters, the stage flags
    the entry with ``pnc_fallback=True`` for downstream handling.

    When ``segments`` are present, each segment is cleaned individually;
    otherwise the top-level entry is cleaned.

    **Alignment update (1st-pass ASR):**  When ``update_alignment`` is
    ``True`` and the segment contains an ``alignment_key`` list (e.g.
    ``"words"``), the cleaned PNC words are written back into the
    alignment entries.  This is used after a 1st-pass ASR where each
    segment carries word-level timestamps.  Set ``cer_threshold=0`` in
    this case so that alignment is only updated when the character
    sequences match exactly.

    Args:
        generation_field:    Key holding the raw LLM output.
        asr_pred_text_key:   Key holding the original ASR text.
        cer_threshold:       Max CER before flagging for BERT PNC fallback.
        punct_marks:         Punctuation characters to strip during comparison.
        vocab_set:           Allowed character set for validity checks (inline string).
        vocab_file:          Optional path to a file containing the vocabulary.
                             If provided, only the file contents are used
                             (``vocab_set`` is ignored).
        segments_key:        Key for the segments list.
        update_alignment:    If ``True``, update word-level alignment entries
                             with the punctuated text when character sequences
                             match.
        alignment_key:       Key within each segment holding the word-level
                             alignment list (each item must have a ``"word"``
                             key).  Defaults to ``"alignment"``.
    """

    generation_field: str = "generation"
    asr_pred_text_key: str = "text"
    cer_threshold: float = 0.01
    punct_marks: str = ".,?"
    vocab_set: str = "abcdefghijklmnopqrstuvwxyz "
    vocab_file: str | None = None
    segments_key: str = "segments"
    update_alignment: bool = False
    alignment_key: str = "alignment"

    name: str = "CleanLLMOutput"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    _cleaned_generation_field: str = field(default="", repr=False)
    _full_vocab_set: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        self._cleaned_generation_field = f"{self.generation_field}_cleaned"

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Load vocabulary file on the worker (avoids file I/O during pickling)."""
        vocab = load_vocab_file(self.vocab_file) if self.vocab_file is not None else set(self.vocab_set)
        self._full_vocab_set = vocab | set(self.punct_marks)

    @staticmethod
    def remove_pncs(text: str, punct_marks: str) -> str:
        text = re.sub(r"[.,?،؟.、？¿!,?।:;]", "", text.lower())  # noqa: RUF001
        pattern = f"[{re.escape(punct_marks)}]"
        text = re.sub(pattern, " ", text.lower())
        return " ".join(text.split())

    @staticmethod
    def clean_llm_output(text: str, punct_marks: str = ".,?") -> str:
        base_remove = r'[](){}<>;:"/\\#*+=&^|~`@“”´»«'  # noqa: RUF001
        removal_set = set(base_remove) - set(punct_marks)
        removal_pattern = "[" + re.escape("".join(removal_set)) + "]"
        text = re.sub(removal_pattern, " ", text)
        text = re.sub(r"\u2014", " ", text)
        text = re.sub(r"-", " ", text)
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"no_think", " ", text)
        text = re.sub(r"The input text is", " ", text)
        pattern = f"([{re.escape(punct_marks)}])\\1+"
        text = re.sub(pattern, r"\1", text)
        return " ".join(text.split())

    @staticmethod
    def is_valid_text(text: str, vocab_set: set[str]) -> bool:
        return all(char in vocab_set for char in text.lower())

    def _process_segment(self, segment: dict) -> None:
        """Clean one segment/entry in place."""
        if self.generation_field not in segment:
            segment["pnc_fallback"] = False
            return
        if self.asr_pred_text_key not in segment:
            segment["pnc_fallback"] = False
            return

        asr_pred_text = segment[self.asr_pred_text_key]
        llm_cleaned = self.clean_llm_output(segment[self.generation_field], self.punct_marks)
        asr_no_pnc = self.remove_pncs(asr_pred_text, self.punct_marks)
        llm_no_pnc = self.remove_pncs(llm_cleaned, self.punct_marks)

        asr_chars = "".join(c for c in asr_no_pnc if c != " ")
        llm_chars = "".join(c for c in llm_no_pnc if c != " ")
        digit_present = any(c.isdigit() for c in llm_no_pnc)

        if asr_chars != llm_chars:
            cer, _, _, _, _ = word_error_rate_detail(
                hypotheses=[asr_no_pnc],
                references=[llm_no_pnc],
                use_cer=True,
            )
            if cer > self.cer_threshold or digit_present or not self.is_valid_text(llm_cleaned, self._full_vocab_set):
                segment["pnc_fallback"] = True
                segment[self._cleaned_generation_field] = asr_pred_text
            else:
                if self.update_alignment and segment.get(self.alignment_key):
                    msg = (
                        "Cannot update alignment when ASR and LLM character "
                        "sequences differ. Set cer_threshold=0 so alignment "
                        "is only updated when texts match exactly."
                    )
                    raise ValueError(msg)
                segment[self._cleaned_generation_field] = llm_cleaned
                segment["pnc_fallback"] = False
        else:
            segment[self._cleaned_generation_field] = llm_cleaned
            segment["pnc_fallback"] = False
            self._update_alignment_words(segment, llm_cleaned)

    def _update_alignment_words(self, segment: dict, cleaned_text: str) -> None:
        """Overwrite alignment word entries with the punctuated text.

        Only acts when ``update_alignment`` is enabled and the segment
        contains an ``alignment_key`` list.  If the word count doesn't
        match the alignment length, the update is skipped with a warning.
        """
        if not self.update_alignment:
            return
        alignment = segment.get(self.alignment_key)
        if not alignment:
            return
        pnc_words = cleaned_text.split()
        non_empty_indices = [i for i, w in enumerate(alignment) if w.get("word", "") != ""]
        if len(pnc_words) != len(non_empty_indices):
            logger.warning(
                f"[{self.name}] PNC word count ({len(pnc_words)}) != alignment "
                f"non-empty entries ({len(non_empty_indices)}); skipping alignment update."
            )
            return
        for word_idx, align_idx in enumerate(non_empty_indices):
            alignment[align_idx]["word"] = pnc_words[word_idx]

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self._cleaned_generation_field, "pnc_fallback"]

    def process(self, task: AudioTask) -> AudioTask:
        data = task.data
        if self.segments_key in data:
            for segment in data[self.segments_key]:
                self._process_segment(segment)
        else:
            self._process_segment(data)
        return task
