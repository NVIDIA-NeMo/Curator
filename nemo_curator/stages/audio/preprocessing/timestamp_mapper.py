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

"""
Timestamp mapper stage.

Resolves segment positions in the concatenated waveform back to
positions in the original audio file using segment mappings stored
in ``task._metadata["segment_mappings"]`` by SegmentConcatenationStage.

Strips waveform from final output items (metadata-only output).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch


def _translate_to_original(
    mappings: List[Dict[str, Any]], concat_start_ms: int, concat_end_ms: int
) -> List[Dict[str, Any]]:
    """Translate concatenated position range to original file positions."""
    results = []
    for m in mappings:
        if m['concat_end_ms'] <= concat_start_ms or m['concat_start_ms'] >= concat_end_ms:
            continue
        overlap_start = max(concat_start_ms, m['concat_start_ms'])
        overlap_end = min(concat_end_ms, m['concat_end_ms'])
        start_offset = overlap_start - m['concat_start_ms']
        end_offset = overlap_end - m['concat_start_ms']
        results.append({
            'original_file': m['original_file'],
            'original_start_ms': m['original_start_ms'] + start_offset,
            'original_end_ms': m['original_start_ms'] + end_offset,
            'duration_ms': end_offset - start_offset,
        })
    return results


@dataclass
class TimestampMapperStage(ProcessingStage[AudioBatch, AudioBatch]):
    """
    Map segment positions back to original file timestamps.

    Reads ``task._metadata["segment_mappings"]`` (written by
    SegmentConcatenationStage) and translates each item's
    ``start_ms`` / ``end_ms`` to ``original_start_ms`` /
    ``original_end_ms`` in the source file.

    Strips ``waveform`` from output items so the final output is
    metadata-only (timestamps, quality scores, speaker info).
    """

    name: str = "TimestampMapper"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self):
        super().__init__()

    def inputs(self) -> Tuple[List[str], List[str]]:
        return ["data"], []

    def outputs(self) -> Tuple[List[str], List[str]]:
        return [], ["original_file", "original_start_ms", "original_end_ms",
                     "duration_ms", "duration_sec"]

    def process(self, task: AudioBatch) -> Optional[AudioBatch]:
        mappings = (task._metadata or {}).get('segment_mappings')

        results: List[Dict[str, Any]] = []

        for item in task.data:
            if mappings:
                concat_start = item.get('start_ms', 0)
                concat_end = item.get('end_ms', 0)
                if concat_end <= concat_start:
                    logger.warning(
                        f"[TimestampMapper] Skipping item with invalid range: "
                        f"start_ms={concat_start}, end_ms={concat_end}"
                    )
                    continue
                original_ranges = _translate_to_original(mappings, concat_start, concat_end)

                for orig in original_ranges:
                    result = self._build_output_item(item, orig)
                    results.append(result)
            else:
                result = self._build_output_item_no_mapping(item)
                results.append(result)

        logger.info(f"[TimestampMapper] {task.task_id}: {len(results)} output segments")

        return AudioBatch(
            data=results,
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            _metadata=task._metadata,
            _stage_perf=list(task._stage_perf),
        )

    @staticmethod
    def _build_output_item(item: Dict[str, Any], orig: Dict[str, Any]) -> Dict[str, Any]:
        """Build final output item from mapped original range."""
        result: Dict[str, Any] = {
            'original_file': orig['original_file'],
            'original_start_ms': orig['original_start_ms'],
            'original_end_ms': orig['original_end_ms'],
            'duration_ms': orig['duration_ms'],
            'duration_sec': orig['duration_ms'] / 1000.0,
        }
        for key in ('speaker_id', 'num_speakers', 'band_prediction',
                     'nisqa_mos', 'nisqa_noi', 'nisqa_col', 'nisqa_dis', 'nisqa_loud',
                     'sigmos_noise', 'sigmos_ovrl', 'sigmos_sig', 'sigmos_col',
                     'sigmos_disc', 'sigmos_loud', 'sigmos_reverb'):
            if key in item and item[key] is not None:
                result[key] = item[key]
        return result

    @staticmethod
    def _build_output_item_no_mapping(item: Dict[str, Any]) -> Dict[str, Any]:
        """Build output item when no segment mappings exist (no concatenation was done)."""
        start_ms = item.get('start_ms', 0)
        end_ms = item.get('end_ms', 0)
        duration_ms = end_ms - start_ms
        result: Dict[str, Any] = {
            'original_file': item.get('original_file', item.get('audio_filepath', 'unknown')),
            'original_start_ms': start_ms,
            'original_end_ms': end_ms,
            'duration_ms': duration_ms,
            'duration_sec': duration_ms / 1000.0,
        }
        for key in ('speaker_id', 'num_speakers', 'band_prediction',
                     'nisqa_mos', 'nisqa_noi', 'nisqa_col', 'nisqa_dis', 'nisqa_loud',
                     'sigmos_noise', 'sigmos_ovrl', 'sigmos_sig', 'sigmos_col',
                     'sigmos_disc', 'sigmos_loud', 'sigmos_reverb'):
            if key in item and item[key] is not None:
                result[key] = item[key]
        return result
