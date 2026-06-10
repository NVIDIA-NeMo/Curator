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

"""IndicVoices ASR dataset handler.

Dataset: ai4bharat/IndicVoices (https://huggingface.co/datasets/ai4bharat/IndicVoices)

The raw dataset is distributed as HuggingFace arrow datasets, one per
``{lang}_{split}`` (e.g. ``hindi_train`` / ``hindi_valid``). Each row carries an
``audio_filepath`` ``Audio`` feature (array + sampling rate), a ``text``
transcript, and assorted metadata (``snr``, ``gender``, ``collectionSource`` ...).

This handler decodes those arrow datasets, converts every clip to
WAV/16 kHz/mono/PCM16 under ``output_dir``, partitions the native ``valid`` split
into ``dev``/``test`` (60/40 by default), and emits one :class:`AudioTask` per
utterance. Manifest writing can either be handled by a downstream writer stage,
or enabled through the base handler's ``write_manifest`` support.

Extraction is parallelized inside a single Xenna worker via ``extraction_workers``.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.stages.audio.asr.datasets.base import BaseASRDatasetHandlerStage
from nemo_curator.stages.audio.asr.metadata import ASRMetadata

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata
    from nemo_curator.tasks import AudioTask, _EmptyTask

# Metadata columns to carry into ASRMetadata.extra when present in a row.
_INDICVOICES_EXTRA_KEYS = (
    "speaker_id",
    "gender",
    "age_group",
    "scenario",
    "task_name",
    "state",
    "district",
    "normalized",
)
_SPLIT_HASH_BUCKETS = 100


@dataclass
class _RowResult:
    meta: ASRMetadata | None
    skip_reason: str | None = None


@dataclass
class IndicVoicesHandler(BaseASRDatasetHandlerStage):
    """Extract the IndicVoices dataset into ASR-training-ready audio tasks.

    Expected input layout depends on ``split_dir_pattern``. The default expects
    one HuggingFace dataset directory per language/split:

    .. code-block:: text

        raw_data_dir/
        ├── gu_train/
        │   ├── data-00000-of-00001.arrow
        │   ├── dataset_info.json
        │   └── state.json
        └── gu_valid/
            ├── data-00000-of-00001.arrow
            ├── dataset_info.json
            └── state.json

    For a downloaded single-language sample like ``raw_data_dir/valid``, pass
    ``split_dir_pattern="{split}"``.

    Output audio is written after split assignment, so native ``train`` remains
    ``train`` while native ``valid`` is deterministically partitioned into
    ``dev`` and ``test``:

    .. code-block:: text

        output_dir/
        └── gu/
            ├── train.jsonl
            ├── dev.jsonl
            ├── test.jsonl
            ├── train/
            │   └── audio/
            │       └── gu_train_0.wav
            ├── dev/
            │   └── audio/
            │       └── gu_valid_0.wav
            └── test/
                └── audio/
                    └── gu_valid_3.wav

    Args:
        native_splits: Native splits to read under ``raw_data_dir``.
        split_dir_pattern: Directory pattern for each native split under
            ``raw_data_dir`` (e.g. ``"{lang}_{split}"`` or ``"{split}"``).
        dev_fraction: Fraction of native ``valid`` assigned to ``dev``;
            the remainder is assigned to ``test``.
    """

    name: str = "indicvoices_handler"
    source_name: str = "IndicVoices"
    native_splits: list[str] = field(default_factory=lambda: ["train", "valid"])
    split_dir_pattern: str = "{lang}_{split}"
    dev_fraction: float = 0.6

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        super().setup(_worker_metadata)
        from datasets import Audio, load_from_disk

        self._load_from_disk = load_from_disk
        self._Audio = Audio

    def _output_splits(self) -> list[str]:
        """Return native output splits after expanding IndicVoices validation data."""
        splits = []
        for native_split in self.native_splits:
            if native_split.lower() in {"valid", "val", "validation"}:
                splits.extend(["dev", "test"])
            else:
                splits.append(native_split)
        return list(dict.fromkeys(splits))

    def coerce_audio(self, audio_obj: Any) -> tuple[Any, int, int]:  # noqa: ANN401
        """Coerce IndicVoices decoded audio into mono ``(array, sample_rate, channels)``.

        ``datasets`` 4.x returns a torchcodec ``AudioDecoder`` while older
        versions return the legacy ``{"array", "sampling_rate"}`` dict.
        """
        np = self._np
        if hasattr(audio_obj, "get_all_samples"):
            samples = audio_obj.get_all_samples()
            arr = samples.data.detach().cpu().numpy()
            sample_rate = int(samples.sample_rate)
        elif isinstance(audio_obj, dict) and "array" in audio_obj:
            arr = np.asarray(audio_obj["array"])
            sample_rate = int(audio_obj["sampling_rate"])
        else:
            msg = f"Unsupported IndicVoices audio object type: {type(audio_obj)!r}"
            raise TypeError(msg)

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            return arr, sample_rate, 1
        if arr.shape[0] <= arr.shape[1]:  # channel-first [C, N]
            return arr.mean(axis=0), sample_rate, int(arr.shape[0])
        return arr.mean(axis=1), sample_rate, int(arr.shape[1])

    def assign_split(self, native_split: str, utt_id: str) -> str:
        """Map an IndicVoices native split to the emitted ``split_type``.

        ``train`` is passed through unchanged. ``valid``/``val``/``validation``
        is split into ``dev`` and ``test`` using a deterministic hash of the
        utterance id, so the same utterance always lands in the same output
        split regardless of worker count or processing order.
        """
        if native_split.lower() in {"valid", "val", "validation"}:
            bucket = int(hashlib.md5(utt_id.encode("utf-8")).hexdigest(), 16) % _SPLIT_HASH_BUCKETS  # noqa: S324
            return "dev" if bucket < self.dev_fraction * _SPLIT_HASH_BUCKETS else "test"
        return native_split

    def _process_row(self, row: dict, index: int, lang: str, native_split: str) -> _RowResult:
        """Convert a single arrow row to WAV + ASRMetadata, including skip reason."""
        if self.text_key not in row or row.get(self.text_key) is None:
            return _RowResult(meta=None, skip_reason="missing_text")
        if self.audio_filepath_key not in row or row.get(self.audio_filepath_key) is None:
            logger.debug(f"[{self.name}] skipping {lang}/{native_split} row {index}: no audio")
            return _RowResult(meta=None, skip_reason="missing_audio")

        utt_id = f"{lang}_{native_split}_{index}"
        split_type = self.assign_split(native_split, utt_id)
        dst_path = os.path.join(self.audio_output_dir(lang, split_type), f"{utt_id}.wav")

        try:
            array, sample_rate, orig_channels = self.coerce_audio(row[self.audio_filepath_key])
            audio_info = self.convert_audio(array, sample_rate, orig_channels, dst_path)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[{self.name}] failed to convert {utt_id}: {e}")
            return _RowResult(meta=None, skip_reason="audio_load")

        extra = {k: row[k] for k in _INDICVOICES_EXTRA_KEYS if k in row}

        return _RowResult(
            meta=ASRMetadata(
                audio_filepath=dst_path,
                text=row[self.text_key],
                duration=audio_info["duration"],
                lang=lang,
                split_type=split_type,
                source=self.source_name,
                sample_rate=self.target_sample_rate,
                num_channels=self.target_channels,
                orig_sample_rate=audio_info["orig_sample_rate"],
                orig_num_channels=audio_info["orig_num_channels"],
                extra=extra,
            )
        )

    def _extract_split(self, lang: str, native_split: str) -> tuple[list[ASRMetadata], dict[str, int]]:
        """Decode one ``{lang}_{split}`` arrow dataset in parallel into ASRMetadata."""
        from joblib import Parallel, delayed

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
        dataset = self._load_from_disk(data_path)
        dataset = dataset.cast_column(self.audio_filepath_key, self._Audio(decode=True))

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
        duration_by_split = dict.fromkeys(["train", "dev", "test", *self._output_splits()], 0.0)
        total_stats = {
            "input_rows": 0,
            "emitted_tasks": 0,
            "skipped_missing_text": 0,
            "skipped_missing_audio": 0,
            "skipped_audio_load": 0,
        }
        for lang in self.langs:
            for native_split in self.native_splits:
                metas, stats = self._extract_split(lang, native_split)
                for key, value in stats.items():
                    total_stats[key] += value
                if not metas:
                    continue
                for meta in metas:
                    duration_by_split[meta.split_type] = duration_by_split.get(meta.split_type, 0.0) + meta.duration
                    self.write_manifest_entry(meta)
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
