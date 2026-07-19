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
from typing import TYPE_CHECKING, Any

import hydra.utils
from loguru import logger

from nemo_curator.stages.audio.inference.asr.adapters.base import ASRAdapter, ASRResult
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata


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


def _set_note(task_data: dict[str, Any], stage_name: str, value: str, notes_key: str) -> None:
    notes = task_data.get(notes_key)
    if not isinstance(notes, dict):
        notes = {}
        task_data[notes_key] = notes
    notes[stage_name] = value


@dataclass
class ASRStage(ProcessingStage[AudioTask, AudioTask]):
    """Audio speech-recognition Curator stage with a pluggable adapter."""

    # Adapter selection.
    adapter_target: str
    model_id: str
    name: str = "ASR_inference"
    revision: str | None = None

    # Task I/O keys.
    waveform_key: str = "waveform"
    sample_rate_key: str = "sampling_rate"
    source_lang_key: str = "source_lang"
    reference_text_key: str | None = None
    default_language: str | None = None
    supported_language_codes: list[str] | None = None
    pred_text_key: str = "pred_text"
    disfluency_text_key: str | None = None
    skip_me_key: str = "_skipme"
    notes_key: str = "additional_notes"
    primary_model_key: str = "primary_model"
    primary_model_value: str | None = None

    skip_if_output_exists: bool = False

    prefetch_fail_on_error: bool = True

    adapter_kwargs: dict[str, Any] = field(default_factory=dict)

    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 32

    def __post_init__(self) -> None:
        if int(self.batch_size) <= 0:
            msg = f"ASRStage.batch_size must be > 0, got {self.batch_size}"
            raise ValueError(msg)
        self.batch_size = int(self.batch_size)
        self._adapter: ASRAdapter | None = None
        self._supported_language_codes = self._normalise_supported_language_codes(self.supported_language_codes)

    @staticmethod
    def _normalise_supported_language_codes(value: object) -> set[str] | None:
        """Normalize an optional adapter-specific supported-language allowlist."""
        if value is None:
            return None
        raw_codes = value.split(",") if isinstance(value, str) else list(value)  # type: ignore[arg-type]
        codes = {str(code).strip().lower() for code in raw_codes if str(code).strip()}
        return codes or None

    def _adapter_class(self) -> type:
        """Resolve the configured adapter lazily to avoid importing optional model dependencies."""
        return hydra.utils.get_class(self.adapter_target)

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        """Cache model weights once per node (no GPU allocation)."""
        try:
            self._adapter_class().prefetch_weights(self.model_id, self.revision)
            logger.info(
                "ASR weights cached on node for {} ({})",
                self.model_id,
                self.adapter_target,
            )
        except Exception as exc:
            msg = f"ASRStage: prefetch_weights failed for {self.model_id}"
            if self.prefetch_fail_on_error:
                raise RuntimeError(msg) from exc
            logger.warning("{}; setup() will retry: {}", msg, exc)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._adapter is None:
            cls = self._adapter_class()
            adapter = cls(
                model_id=self.model_id,
                revision=self.revision,
                **self.adapter_kwargs,
            )
            try:
                adapter.setup()
            except Exception:
                try:
                    adapter.teardown()
                except Exception as teardown_exc:  # noqa: BLE001
                    logger.warning("ASR adapter cleanup after setup failure also failed: {}", teardown_exc)
                raise
            self._adapter = adapter
            logger.info("ASR adapter ready on worker ({})", self.adapter_target)

    def teardown(self) -> None:
        if self._adapter is not None:
            self._adapter.teardown()
            self._adapter = None

    def inputs(self) -> tuple[list[str], list[str]]:
        optional_inputs = [self.waveform_key, self.sample_rate_key]
        if self.reference_text_key:
            optional_inputs.append(self.reference_text_key)
        return [], optional_inputs

    def outputs(self) -> tuple[list[str], list[str]]:
        keys = [self.pred_text_key]
        if self.disfluency_text_key:
            keys.append(self.disfluency_text_key)
        return [], keys

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

    def _resolve_reference_text(self, task: AudioTask) -> str | None:
        if not self.reference_text_key:
            return None
        value = task.data.get(self.reference_text_key)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _build_items(self, tasks: list[AudioTask]) -> list[dict[str, Any]]:
        items = []
        for task in tasks:
            waveform = task.data.get(self.waveform_key)
            sample_rate = task.data.get(self.sample_rate_key)
            items.append(
                {
                    "waveform": waveform,
                    "sample_rate": sample_rate,
                    "language": self._resolve_language(task),
                    "language_code": self._resolve_language_code(task),
                    "reference_text": self._resolve_reference_text(task),
                    "task_id": task.task_id,
                }
            )
        return items

    def process(self, task: AudioTask) -> AudioTask:
        msg = f"{type(self).__name__} only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Run one ASR batch."""
        if len(tasks) == 0:
            return []
        return self._process_plain_batch(tasks)

    def _partition_inference_tasks(self, tasks: list[AudioTask]) -> tuple[list[AudioTask], int]:
        tasks_to_process: list[AudioTask] = []
        output_exists_skipped = 0
        for task in tasks:
            if self.skip_if_output_exists and task.data.get(self.pred_text_key):
                output_exists_skipped += 1
                task.data.pop(self.waveform_key, None)
                continue
            tasks_to_process.append(task)
        return tasks_to_process, output_exists_skipped

    def _process_plain_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Dispatch one backend batch."""
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
        if output_exists_skipped:
            logger.info(
                "ASRStage ({}): reused existing {} for {}/{} tasks",
                self.adapter_target,
                self.pred_text_key,
                output_exists_skipped,
                len(tasks),
            )
        return tasks

    def run_inference(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        """Transcribe one capped sub-batch via the adapter."""
        supported_indices = [index for index, item in enumerate(items) if self._is_language_supported(item)]
        by_index: dict[int, ASRResult] = {}
        if supported_indices:
            supported_items = [items[index] for index in supported_indices]
            adapter_results = self._adapter.transcribe_batch(supported_items)
            if len(adapter_results) != len(supported_items):
                msg = (
                    f"Adapter returned {len(adapter_results)} results for "
                    f"{len(supported_items)} supported items (must match 1:1)"
                )
                raise RuntimeError(msg)
            by_index = dict(zip(supported_indices, adapter_results, strict=True))
        return [
            by_index.get(
                index,
                ASRResult(
                    text="",
                    skipped=False,
                    extras={"unsupported_language": str(item.get("language_code", "") or "").strip().lower()},
                ),
            )
            for index, item in enumerate(items)
        ]

    def assemble(
        self,
        tasks: list[AudioTask],
        _items: list[dict[str, Any]],
        results: list[ASRResult],
    ) -> list[AudioTask]:
        """Write adapter results to tasks."""
        skipped_count = 0
        for task, result in zip(tasks, results, strict=True):
            task.data[self.pred_text_key] = result.text
            if self.disfluency_text_key:
                task.data[self.disfluency_text_key] = result.secondary_text or ""
            unsupported_language = result.extras.get("unsupported_language")
            if unsupported_language:
                _set_note(
                    task.data,
                    self.name,
                    f"skipped (unsupported language: {unsupported_language})",
                    self.notes_key,
                )
                _set_note(
                    task.data,
                    self.pred_text_key,
                    f"lang_not_supported:{unsupported_language}",
                    self.notes_key,
                )
            if result.skipped:
                task.data[self.skip_me_key] = str(result.extras.get("skip_reason") or "empty_audio")
                skipped_count += 1
            if self.primary_model_value and not unsupported_language:
                _set_note(task.data, self.primary_model_key, self.primary_model_value, self.notes_key)
            task.data.pop(self.waveform_key, None)

        if skipped_count:
            logger.info(
                f"ASRStage ({self.adapter_target}): marked {skipped_count}/{len(tasks)} "
                f"tasks as empty_audio ({self.skip_me_key})",
            )
        logger.debug(
            f"ASRStage ({self.adapter_target}): generated {len(results)} predictions "
            f"(disfluency_text={'on' if self.disfluency_text_key else 'off'})",
        )
        return tasks
