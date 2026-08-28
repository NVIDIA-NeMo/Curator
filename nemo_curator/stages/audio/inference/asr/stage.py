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

"""Generic audio ASR Curator stage with a pluggable adapter.

Curator-side glue validates I/O, resolves per-task language, and writes
predictions. The concrete adapter is resolved at runtime from
``adapter_target`` via ``hydra.utils.get_class``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torchaudio
from loguru import logger

from nemo_curator.models.asr.base import ASRAdapter, ASRResult
from nemo_curator.stages.audio.inference.base import AdapterInferenceStage
from nemo_curator.stages.audio.model_input_segmentation import (
    plan_audio_segments,
    resolve_max_model_input_duration,
)
from nemo_curator.stages.resources import Resources

if TYPE_CHECKING:
    from nemo_curator.stages.audio.inference.batch_policy import BatchPolicy
    from nemo_curator.tasks import AudioTask


# ISO code -> human-readable name; the adapter receives the resolved name.
_LANG_CODE_TO_NAME: dict[str, str] = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "mt": "Maltese",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}

_SKIP_ME_KEY = "_skipme"
_NOTES_KEY = "additional_notes"
_MONO_DIMENSIONS = 1
_CHANNEL_FIRST_DIMENSIONS = 2


def _set_note(task_data: dict[str, Any], stage_name: str, value: str) -> None:
    notes = task_data.get(_NOTES_KEY)
    if not isinstance(notes, dict):
        notes = {}
        task_data[_NOTES_KEY] = notes
    notes[stage_name] = value


@dataclass
class ASRStage(AdapterInferenceStage[ASRAdapter]):
    """Audio speech-recognition stage with a pluggable adapter.

    The stage writes ``pred_text_key`` and optional control columns ``_skipme``
    and ``additional_notes``. When ``extras_key`` is configured, it also writes
    non-empty adapter metadata as one nested dictionary under that key.

    Audio longer than ``max_inference_duration_s`` is first split into
    model-safe chunks and stitched back to one result per parent row. An
    optional ``batch_policy`` then locally re-partitions those chunks in each
    backend-provided ``process_batch`` call into duration/cost-coherent adapter
    calls. It does not change backend scheduling or move batching across worker
    calls.
    """

    # Adapter selection.
    adapter_target: str
    model_id: str
    name: str = "ASR_inference"

    # Task I/O keys.
    audio_filepath_key: str = "resampled_audio_filepath"
    waveform_key: str | None = None
    sample_rate_key: str = "sampling_rate"
    target_sample_rate: int = 16000
    keep_waveform: bool = False
    source_lang_key: str = "source_lang"
    default_language: str | None = None
    supported_language_codes: list[str] | None = None
    pred_text_key: str = "pred_text"
    extras_key: str | None = None

    skip_if_output_exists: bool = False
    fail_on_audio_error: bool = False

    prefetch_fail_on_error: bool = True

    adapter_kwargs: dict[str, Any] = field(default_factory=dict)

    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 32
    max_inference_duration_s: float = 2400.0
    adapter_batch_size: int | None = None
    batch_policy: BatchPolicy | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.pred_text_key:
            msg = "ASRStage.pred_text_key must be non-empty"
            raise ValueError(msg)
        if self.pred_text_key in {_SKIP_ME_KEY, _NOTES_KEY}:
            msg = f"ASRStage.pred_text_key cannot use reserved control column {self.pred_text_key!r}"
            raise ValueError(msg)
        if self.extras_key is not None:
            self.extras_key = self.extras_key.strip()
            if not self.extras_key:
                msg = "ASRStage.extras_key must be non-empty or None"
                raise ValueError(msg)
            if self.extras_key in {self.pred_text_key, _SKIP_ME_KEY, _NOTES_KEY}:
                msg = f"ASRStage.extras_key cannot collide with another output column: {self.extras_key!r}"
                raise ValueError(msg)
        if int(self.batch_size) <= 0:
            msg = f"ASRStage.batch_size must be > 0, got {self.batch_size}"
            raise ValueError(msg)
        self.adapter_batch_size = self._validate_adapter_batch_size(self.adapter_batch_size)
        if int(self.target_sample_rate) <= 0:
            msg = f"ASRStage.target_sample_rate must be > 0, got {self.target_sample_rate}"
            raise ValueError(msg)
        self.max_inference_duration_s = resolve_max_model_input_duration(
            max_duration_s=self.max_inference_duration_s,
            owner="ASRStage",
        )
        self.batch_size = int(self.batch_size)
        self.target_sample_rate = int(self.target_sample_rate)
        self._supported_language_codes = self._normalise_supported_language_codes(self.supported_language_codes)

    @staticmethod
    def _validate_adapter_batch_size(value: int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            msg = f"ASRStage.adapter_batch_size must be an int or None, got {type(value).__name__}"
            raise TypeError(msg)
        if value <= 0:
            msg = f"ASRStage.adapter_batch_size must be > 0, got {value}"
            raise ValueError(msg)
        return value

    @staticmethod
    def _normalise_supported_language_codes(value: object) -> set[str] | None:
        """Normalize an optional adapter-specific supported-language allowlist."""
        if value is None:
            return None
        raw_codes = value.split(",") if isinstance(value, str) else list(value)  # type: ignore[arg-type]
        codes = {str(code).strip().lower() for code in raw_codes if str(code).strip()}
        return codes or None

    def _create_adapter(self) -> ASRAdapter:
        """Construct the configured ASR adapter."""
        adapter_cls = self._adapter_class()
        return cast(
            "ASRAdapter",
            adapter_cls(
                model_id=self.model_id,
                **self.adapter_kwargs,
            ),
        )

    def outputs(self) -> tuple[list[str], list[str]]:
        optional_outputs = [self.pred_text_key, _SKIP_ME_KEY, _NOTES_KEY]
        if self.extras_key is not None:
            optional_outputs.append(self.extras_key)
        return [], optional_outputs

    def _resolve_language(self, task: AudioTask) -> str | None:
        code = self._resolve_language_code(task)
        if code:
            return _LANG_CODE_TO_NAME.get(code, code)
        return None

    def _resolve_language_code(self, task: AudioTask) -> str | None:
        code = task.data.get(self.source_lang_key) if self.source_lang_key else None
        if code:
            return str(code).strip().lower()
        if self.default_language:
            return str(self.default_language).strip().lower()
        return None

    def _is_language_supported(self, item: dict[str, Any]) -> bool:
        if self._supported_language_codes is None:
            return True
        code = str(item.get("language_code", "") or "").strip().lower()
        return bool(code) and code in self._supported_language_codes

    def _build_items(self, tasks: list[AudioTask]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for task in tasks:
            item = {
                "language": self._resolve_language(task),
                "language_code": self._resolve_language_code(task),
                "task_id": task.task_id,
            }
            if self.waveform_key:
                item["waveform"] = task.data[self.waveform_key]
                item["sample_rate"] = task.data[self.sample_rate_key]
            else:
                item["audio_filepath"] = task.data[self.audio_filepath_key]
            items.append(item)
        return items

    @staticmethod
    def _load_audio(audio_filepath: str) -> tuple[np.ndarray, int]:
        """Open one resampled file inside the ASR worker.

        ``torchaudio.load`` returns channel-first audio. Resampled pipeline
        inputs are normally mono, so squeezing removes that singleton channel;
        multichannel inputs remain channel-first for ``_prepare_waveform`` to
        downmix.
        """
        waveform, sample_rate = torchaudio.load(audio_filepath)
        return waveform.squeeze(0).numpy(), sample_rate

    def _prepare_waveform(self, waveform: object, sample_rate: object) -> np.ndarray:
        """Return contiguous mono float32 samples at ``target_sample_rate``."""
        source_sample_rate = int(sample_rate)
        if source_sample_rate <= 0:
            msg = f"sample rate must be > 0, got {source_sample_rate}"
            raise ValueError(msg)

        tensor = torch.as_tensor(waveform, dtype=torch.float32)
        if tensor.ndim == _CHANNEL_FIRST_DIMENSIONS:
            tensor = tensor.mean(dim=0)
        elif tensor.ndim != _MONO_DIMENSIONS:
            msg = f"waveform must be 1-D mono or 2-D channel-first audio, got shape {tuple(tensor.shape)}"
            raise ValueError(msg)
        if source_sample_rate != self.target_sample_rate:
            tensor = torchaudio.functional.resample(
                tensor,
                source_sample_rate,
                self.target_sample_rate,
            )
        return np.ascontiguousarray(tensor.cpu().numpy(), dtype=np.float32)

    def process(self, task: AudioTask) -> AudioTask:
        msg = f"{type(self).__name__} only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Run one ASR batch."""
        tasks_to_process, output_exists_skipped = self._partition_inference_tasks(tasks)

        for task in tasks_to_process:
            if not self.validate_input(task):
                msg = f"Task {task.task_id} missing required columns for {type(self).__name__}: {self.inputs()}"
                raise ValueError(msg)
        if self._adapter is None:
            msg = "Adapter not initialized - setup() was not called"
            raise RuntimeError(msg)

        items = self._build_items(tasks_to_process)

        results = self.run_inference(items)
        if len(results) != len(items):
            msg = f"run_fn returned {len(results)} results for {len(items)} items (must match 1:1)"
            raise RuntimeError(msg)
        self.assemble(
            tasks_to_process,
            items,
            results,
        )
        if self.waveform_key and not self.keep_waveform:
            for task in tasks:
                task.data.pop(self.waveform_key, None)
        if output_exists_skipped:
            logger.info(
                "ASRStage ({}): reused existing {} for {}/{} tasks",
                self.adapter_target,
                self.pred_text_key,
                output_exists_skipped,
                len(tasks),
            )
        return tasks

    def _partition_inference_tasks(self, tasks: list[AudioTask]) -> tuple[list[AudioTask], int]:
        tasks_to_process: list[AudioTask] = []
        output_exists_skipped = 0
        for task in tasks:
            if self.skip_if_output_exists and task.data.get(self.pred_text_key):
                output_exists_skipped += 1
                continue
            tasks_to_process.append(task)
        return tasks_to_process, output_exists_skipped

    def run_inference(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        """Transcribe one stage batch via the adapter."""
        supported_indices = [index for index, item in enumerate(items) if self._is_language_supported(item)]
        by_index: dict[int, ASRResult] = {}
        adapter_parent_indices: list[int] = []
        adapter_items: list[dict[str, Any]] = []
        for index in supported_indices:
            item = items[index]
            try:
                if "waveform" in item:
                    waveform = item["waveform"]
                    sample_rate = item["sample_rate"]
                    audio_source = self.waveform_key or "waveform"
                else:
                    audio_source = str(item["audio_filepath"])
                    waveform, sample_rate = self._load_audio(audio_source)
                waveform = self._prepare_waveform(waveform, sample_rate)
            except Exception as exc:
                if self.fail_on_audio_error:
                    msg = f"ASRStage ({self.adapter_target}): failed to prepare audio for task {item['task_id']} from {audio_source}"
                    raise RuntimeError(msg) from exc
                logger.warning(
                    "ASRStage ({}): failed to prepare audio for task {} from {}: {}",
                    self.adapter_target,
                    item["task_id"],
                    audio_source,
                    exc,
                )
                by_index[index] = ASRResult(text="", skipped=True, skip_reason="audio_load_error")
                continue
            segments = plan_audio_segments(
                num_samples=int(waveform.shape[0]),
                sample_rate=self.target_sample_rate,
                max_duration_s=self.max_inference_duration_s,
                owner="ASRStage",
            )
            for segment in segments:
                adapter_parent_indices.append(index)
                adapter_items.append(
                    {
                        "waveform": np.ascontiguousarray(
                            waveform[segment.start_sample : segment.stop_sample],
                            dtype=np.float32,
                        ),
                        "sample_rate": self.target_sample_rate,
                        "audio_seconds": segment.duration_s,
                        "language": item["language"],
                        "language_code": item["language_code"],
                        "task_id": item["task_id"],
                        "chunk_idx": segment.index,
                        "chunk_count": segment.count,
                    }
                )

        if adapter_items:
            adapter_results = self._run_adapter_batches(adapter_items)
            per_parent: dict[int, list[ASRResult]] = {}
            for parent_index, result in zip(adapter_parent_indices, adapter_results, strict=True):
                per_parent.setdefault(parent_index, []).append(result)
            for parent_index, chunk_results in per_parent.items():
                by_index[parent_index] = self._stitch_chunk_results(chunk_results)
        return [
            by_index.get(
                index,
                ASRResult(
                    text="",
                    skipped=True,
                    skip_reason=(
                        "language_not_supported"
                        if str(item.get("language_code", "") or "").strip()
                        else "language_missing"
                    ),
                    unsupported_language=str(item.get("language_code", "") or "").strip().lower() or None,
                ),
            )
            for index, item in enumerate(items)
        ]

    @staticmethod
    def _stitch_chunk_results(results: list[ASRResult]) -> ASRResult:
        """Join ordered chunk outputs into one parent-row result."""
        if not results:
            return ASRResult(text="", skipped=True, skip_reason="empty_audio")
        if len(results) == 1:
            return results[0]

        texts = [text for result in results if (text := (result.text or "").strip())]
        all_skipped = all(result.skipped for result in results)
        skip_reason = next((result.skip_reason for result in results if result.skip_reason), None)
        unsupported_language = next(
            (result.unsupported_language for result in results if result.unsupported_language),
            None,
        )
        chunk_extras = [dict(result.extras) for result in results]
        extras = {"chunks": chunk_extras} if any(chunk_extras) else {}
        return ASRResult(
            text=" ".join(texts),
            skipped=all_skipped,
            skip_reason=skip_reason if all_skipped else None,
            unsupported_language=unsupported_language,
            extras=extras,
        )

    def _run_adapter_batches(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        """Run locally planned adapter calls and restore candidate-row order."""
        if self._adapter is None:
            msg = "Adapter not initialized - setup() was not called"
            raise RuntimeError(msg)

        policy = self.batch_policy
        if policy is not None and policy.enabled:
            sub_batches = policy.bucketize(items, cost_fn=self.item_cost)
        else:
            cap_source = self.adapter_batch_size if self.adapter_batch_size is not None else self.batch_size
            cap = max(1, int(cap_source))
            sub_batches = [
                (list(range(start, min(start + cap, len(items)))), items[start : start + cap])
                for start in range(0, len(items), cap)
            ]

        aligned: list[ASRResult | None] = [None] * len(items)
        for indices, sub_items in sub_batches:
            sub_results = self._adapter.transcribe_batch(sub_items)
            if len(sub_results) != len(sub_items):
                msg = (
                    f"Adapter returned {len(sub_results)} results for "
                    f"{len(sub_items)} supported items (must match 1:1)"
                )
                raise RuntimeError(msg)
            for index, result in zip(indices, sub_results, strict=True):
                aligned[index] = result

        if any(result is None for result in aligned):
            msg = "Local batch planning did not produce a result for every supported item"
            raise RuntimeError(msg)
        return [result for result in aligned if result is not None]

    def item_cost(self, item: dict[str, Any]) -> float:
        """Return the local bucketing cost for one prepared adapter item."""
        estimator = getattr(self._adapter, "estimate_item_cost", None)
        if callable(estimator):
            try:
                estimated = estimator(item)
                if isinstance(estimated, Real):
                    return max(0.0, float(estimated))
            except Exception as exc:  # noqa: BLE001
                logger.debug("ASR adapter cost estimator failed; falling back to duration cost: {}", exc)
        for key in ("estimated_vram_units", "estimated_encoder_tokens"):
            value = item.get(key)
            if value is not None:
                return max(0.0, float(value))
        return max(0.0, float(item.get("audio_seconds", 0.0)))

    def assemble(
        self,
        tasks: list[AudioTask],
        items: list[dict[str, Any]],
        results: list[ASRResult],
    ) -> list[AudioTask]:
        """Write adapter results to tasks."""
        skipped_count = 0
        for task, item, result in zip(tasks, items, results, strict=True):
            task.data[self.pred_text_key] = result.text
            if self.extras_key is not None:
                if result.extras:
                    task.data[self.extras_key] = dict(result.extras)
                else:
                    task.data.pop(self.extras_key, None)
            unsupported_language = result.unsupported_language
            missing_language = self._supported_language_codes is not None and not item["language_code"]
            if missing_language:
                _set_note(task.data, self.name, "skipped (missing language)")
                _set_note(task.data, self.pred_text_key, "language_missing")
            elif unsupported_language:
                _set_note(
                    task.data,
                    self.name,
                    f"skipped (unsupported language: {unsupported_language})",
                )
                _set_note(
                    task.data,
                    self.pred_text_key,
                    f"lang_not_supported:{unsupported_language}",
                )
            if result.skipped:
                task.data[_SKIP_ME_KEY] = result.skip_reason or "empty_audio"
                skipped_count += 1

        if skipped_count:
            logger.info(
                f"ASRStage ({self.adapter_target}): marked {skipped_count}/{len(tasks)} tasks with {_SKIP_ME_KEY}",
            )
        logger.debug(
            f"ASRStage ({self.adapter_target}): generated {len(results)} predictions",
        )
        return tasks
