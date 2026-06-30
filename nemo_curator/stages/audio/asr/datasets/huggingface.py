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

"""Generic Hugging Face ASR dataset handler for saved Arrow datasets."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from datasets import Audio, load_from_disk
from joblib import Parallel, delayed
from loguru import logger

from nemo_curator.stages.audio.asr.datasets.base import BaseASRDatasetHandlerStage
from nemo_curator.stages.audio.asr.metadata import ASRMetadata

if TYPE_CHECKING:
    from nemo_curator.tasks import AudioTask, _EmptyTask

_SOURCE_FIELD_MAPPING_BY_SOURCE = {
    "indicvoices": {
        "speaker_id": "speaker_id",
        "gender": "gender",
        "age_group": "age",
        "scenario": "scenario",
        "task_name": "task_name",
        "state": "state",
        "district": "district",
        "normalized": "normalized",
    },
    "kathbath": {
        "fname": "fname",
        "speaker_id": "speaker_id",
        "gender": "gender",
    },
    "shrutilipi": {},
}
_SUPPORTED_SOURCE_NAMES = {
    source_name.lower(): source_name for source_name in ("IndicVoices", "Kathbath", "Shrutilipi")
}
_ASR_METADATA_FIELD_NAMES = {metadata_field.name for metadata_field in fields(ASRMetadata)} - {"extra"}


@dataclass
class _RowResult:
    meta: ASRMetadata | None
    skip_reason: str | None = None


@dataclass
class HuggingFaceASRDatasetHandler(BaseASRDatasetHandlerStage):
    """Extract saved Hugging Face ASR datasets into canonical ASR audio tasks.

    The handler expects datasets that were written with ``Dataset.save_to_disk``
    and contain an audio column compatible with ``datasets.Audio``.
    """

    name: str = "huggingface_asr_dataset_handler"
    native_splits: list[str] = field(default_factory=lambda: ["train", "valid"])
    split_dir_pattern: str = "{lang}/{split}"
    valid_split_strategy: Literal["keep", "map", "dev_test"] = "keep"
    dev_fraction: float = 0.6
    hash_buckets: int = 100
    split_mapping: dict[str, str] | None = None
    duration_key: str | None = "duration"
    filename_key: str | None = "fname"
    extra_keys: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.source_name.lower() not in _SUPPORTED_SOURCE_NAMES:
            supported_sources = ", ".join(_SUPPORTED_SOURCE_NAMES.values())
            msg = (
                f"Unsupported source_name '{self.source_name}' for {type(self).__name__}. "
                f"Supported source names: {supported_sources}"
            )
            raise ValueError(msg)

    def _output_splits(self) -> list[str]:
        if self.manifest_splits:
            return list(dict.fromkeys(self.manifest_splits))
        splits = []
        for native_split in self.native_splits:
            if self._is_validation_split(native_split) and self.valid_split_strategy == "dev_test":
                splits.extend(["dev", "test"])
            else:
                splits.append(self.assign_split(native_split))
        return list(dict.fromkeys(splits))

    def assign_split(self, native_split: str, utt_id: str | None = None) -> str:
        """Map native dataset split names to emitted split names."""
        if self.split_mapping and native_split in self.split_mapping:
            return self.split_mapping[native_split]
        if self._is_validation_split(native_split) and self.valid_split_strategy == "dev_test":
            if utt_id is None:
                msg = "utt_id is required when valid_split_strategy='dev_test'"
                raise ValueError(msg)
            bucket = int(hashlib.md5(utt_id.encode("utf-8")).hexdigest(), 16) % self.hash_buckets  # noqa: S324
            return "dev" if bucket < self.dev_fraction * self.hash_buckets else "test"
        return native_split

    @staticmethod
    def _is_validation_split(native_split: str) -> bool:
        return native_split.lower() in {"valid", "val", "validation"}

    def _source_field_mapping(self) -> dict[str, str]:
        if self.extra_keys:
            return {key: key for key in self.extra_keys}
        return _SOURCE_FIELD_MAPPING_BY_SOURCE[self.source_name.lower()]

    def _metadata_fields_from_row(self, row: dict) -> tuple[dict[str, object], dict[str, object]]:
        metadata_fields = {}
        extra = {}
        for source_key, output_key in self._source_field_mapping().items():
            if source_key not in row:
                continue
            if output_key in _ASR_METADATA_FIELD_NAMES:
                metadata_fields[output_key] = row[source_key]
            else:
                extra[output_key] = row[source_key]
        return metadata_fields, extra

    def coerce_audio(self, audio_obj: Any) -> tuple[Any, int, int]:  # noqa: ANN401
        """Coerce decoded Hugging Face audio into mono ``(array, sample_rate, channels)``."""
        np = self._np
        if hasattr(audio_obj, "get_all_samples"):
            samples = audio_obj.get_all_samples()
            arr = samples.data.detach().cpu().numpy()
            sample_rate = int(samples.sample_rate)
        elif isinstance(audio_obj, dict) and "array" in audio_obj:
            arr = np.asarray(audio_obj["array"])
            sample_rate = int(audio_obj["sampling_rate"])
        else:
            msg = f"Unsupported Hugging Face audio object type: {type(audio_obj)!r}"
            raise TypeError(msg)

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            return arr, sample_rate, 1
        if arr.shape[0] <= arr.shape[1]:
            return arr.mean(axis=0), sample_rate, int(arr.shape[0])
        return arr.mean(axis=1), sample_rate, int(arr.shape[1])

    def _audio_filename(self, row: dict, utt_id: str) -> str:
        if self.filename_key and row.get(self.filename_key):
            return f"{Path(str(row[self.filename_key])).stem}.wav"
        return f"{utt_id}.wav"

    def _process_row(self, row: dict, index: int, lang: str, native_split: str) -> _RowResult:
        if self.text_key not in row or row.get(self.text_key) is None:
            return _RowResult(meta=None, skip_reason="missing_text")
        if self.audio_filepath_key not in row or row.get(self.audio_filepath_key) is None:
            logger.debug(f"[{self.name}] skipping {lang}/{native_split} row {index}: no audio")
            return _RowResult(meta=None, skip_reason="missing_audio")

        utt_id = f"{lang}_{native_split}_{index}"
        split_type = self.assign_split(native_split, utt_id)
        dst_path = os.path.join(self.audio_output_dir(lang, split_type), self._audio_filename(row, utt_id))

        try:
            array, sample_rate, orig_channels = self.coerce_audio(row[self.audio_filepath_key])
            audio_info = self.convert_audio(array, sample_rate, orig_channels, dst_path)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[{self.name}] failed to convert {utt_id}: {e}")
            return _RowResult(meta=None, skip_reason="audio_load")

        metadata_fields, extra = self._metadata_fields_from_row(row)
        return _RowResult(
            meta=ASRMetadata(
                audio_filepath=dst_path,
                text=str(row[self.text_key]),
                duration=audio_info["duration"],
                lang=lang,
                split_type=split_type,
                source=self.source_name,
                sample_rate=self.target_sample_rate,
                num_channels=self.target_channels,
                orig_sample_rate=audio_info["orig_sample_rate"],
                orig_num_channels=audio_info["orig_num_channels"],
                **metadata_fields,
                extra=extra,
            )
        )

    def _extract_split(self, lang: str, native_split: str) -> tuple[list[ASRMetadata], dict[str, int]]:
        stats = {
            "input_rows": 0,
            "emitted_tasks": 0,
            "skipped_missing_text": 0,
            "skipped_missing_audio": 0,
            "skipped_audio_load": 0,
        }
        data_path = os.path.join(self.raw_data_dir, self.split_dir_pattern.format(lang=lang, split=native_split))
        if not os.path.isdir(data_path):
            logger.warning(f"[{self.name}] missing dataset dir, skipping: {data_path}")
            return [], stats

        logger.info(f"[{self.name}] loading {data_path}")
        dataset = load_from_disk(data_path)
        dataset = dataset.cast_column(self.audio_filepath_key, Audio(decode=True))

        def load_and_process(index: int) -> _RowResult:
            try:
                row = dataset[index]
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[{self.name}] failed to load row {lang}/{native_split}/{index}: {e}")
                return _RowResult(meta=None, skip_reason="audio_load")
            return self._process_row(row, index, lang, native_split)

        start = time.perf_counter()
        results = Parallel(n_jobs=self.extraction_workers, backend="threading")(
            delayed(load_and_process)(i) for i in range(len(dataset))
        )
        metas = [result.meta for result in results if result.meta is not None]
        for result in results:
            if result.skip_reason:
                stats[f"skipped_{result.skip_reason}"] += 1
        stats["input_rows"] = len(results)
        stats["emitted_tasks"] = len(metas)
        logger.info(
            f"[{self.name}] {lang}/{native_split}: extracted {len(metas)}/{len(results)} "
            f"(missing_text={stats['skipped_missing_text']}, missing_audio={stats['skipped_missing_audio']}, "
            f"audio_load_failed={stats['skipped_audio_load']}) "
            f"in {time.perf_counter() - start:.1f}s"
        )
        return metas, stats

    def process(self, _: _EmptyTask) -> list[AudioTask]:
        start = time.perf_counter()
        all_tasks: list[AudioTask] = []
        total_stats = {
            "input_rows": 0,
            "emitted_tasks": 0,
            "skipped_missing_text": 0,
            "skipped_missing_audio": 0,
            "skipped_audio_load": 0,
        }
        duration_by_split = dict.fromkeys(["train", "dev", "test", *self._output_splits()], 0.0)
        for lang in self.langs:
            for native_split in self.native_splits:
                metas, stats = self._extract_split(lang, native_split)
                for key, value in stats.items():
                    total_stats[key] += value
                for meta in metas:
                    duration_by_split[meta.split_type] = duration_by_split.get(meta.split_type, 0.0) + meta.duration
                    self.write_manifest_entry(meta)
                # multiple languages can be processed in one go if we are not storing tasks in memory.
                if not self.write_manifest:
                    all_tasks.extend(self.build_audio_task(meta) for meta in metas)
        total_stats["emitted_tasks"] = len(all_tasks)
        for split_type, duration_seconds in duration_by_split.items():
            total_stats[f"duration_{split_type}_seconds"] = duration_seconds
            total_stats[f"duration_{split_type}_hours"] = duration_seconds / 3600
        total_stats["process_time"] = time.perf_counter() - start
        self._log_metrics(total_stats)
        duration_summary = ", ".join(
            f"{split_type}={duration_by_split.get(split_type, 0.0) / 3600:.2f}h"
            for split_type in ["train", "dev", "test"]
        )
        logger.info(
            f"[{self.name}] emitted {len(all_tasks)} AudioTasks "
            f"(input_rows={total_stats['input_rows']}, skipped_missing_text={total_stats['skipped_missing_text']}, "
            f"skipped_missing_audio={total_stats['skipped_missing_audio']}, "
            f"skipped_audio_load={total_stats['skipped_audio_load']}, duration_by_split_hours=({duration_summary}))"
        )
        return all_tasks
