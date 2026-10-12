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

"""Judging stage: keep only clips whose audio really says the intended number.

Text-to-speech systems misread some numbers, most often long ones. The judges here never see the
target. They transcribe the clip and the verdict is a programmatic value comparison of the
transcript with the target, so a judge cannot be talked into "hearing" a number it was told to
expect, which is a known failure mode of judges that are shown the expected answer in their prompt.

- ``blind``: one audio-capable LLM transcribes; faithful iff the transcript's value matches.
- ``consensus``: several ASR/LLM backends transcribe; faithful iff at least ``min_agree`` of them
  match. With ``min_agree=1`` an exact multi-digit match against a wrong rendering is implausible,
  so any match certifies the audio and independent failure modes keep false accepts near zero.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.audio.numpipe.asr import make_asr_backend
from nemo_curator.stages.audio.numpipe.numeric_norm import numeric_eq
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


class Judge(Protocol):
    def judge(self, wav_path: str | Path, number: str) -> dict[str, Any]: ...


@dataclass
class BlindJudge:
    """Transcribe with one LLM backend (never told the target), then compare values.

    ``base_url`` is an OpenAI-compatible chat endpoint (ending in ``/v1``) and ``model`` an
    audio-capable model there.
    """

    model: str
    base_url: str
    api_key_env: str | None = None
    homophones: bool = True

    def __post_init__(self) -> None:
        cfg: dict[str, Any] = {"backend": "llm", "model": self.model, "base_url": self.base_url}
        if self.api_key_env:
            cfg["api_key_env"] = self.api_key_env
        self._backend = make_asr_backend(cfg)

    def judge(self, wav_path: str | Path, number: str) -> dict[str, Any]:
        heard = self._backend.transcribe(wav_path)
        return {
            "faithful": numeric_eq(heard, number, homophones=self.homophones),
            "heard": heard,
            "reason": "blind transcribe-then-compare",
            "method": "blind",
        }


@dataclass
class ConsensusJudge:
    """Several transcribers vote by value match; ``min_agree`` matches make the clip faithful."""

    backends: list[dict[str, Any]]
    min_agree: int = 1
    homophones: bool = True

    def __post_init__(self) -> None:
        if not self.backends:
            msg = "consensus judge needs at least one ASR backend config"
            raise ValueError(msg)
        self._voters = [make_asr_backend(cfg) for cfg in self.backends]

    def transcripts(self, wav_path: str | Path) -> list[str]:
        out: list[str] = []
        for voter in self._voters:
            try:
                out.append(voter.transcribe(wav_path))
            except Exception as exc:  # noqa: BLE001 - a failing voter abstains instead of blocking the clip
                out.append(f"<error: {exc}>")
        return out

    def judge(self, wav_path: str | Path, number: str) -> dict[str, Any]:
        transcripts = self.transcripts(wav_path)
        votes = sum(numeric_eq(t, number, homophones=self.homophones) for t in transcripts)
        return {
            "faithful": votes >= self.min_agree,
            "heard": " | ".join(transcripts),
            "reason": f"consensus {votes}/{len(transcripts)} (need {self.min_agree})",
            "method": "consensus",
        }


JUDGES: dict[str, type] = {"blind": BlindJudge, "consensus": ConsensusJudge}


def make_judge(config: dict[str, Any]) -> Judge:
    """Build a judge from ``{"method": <name>, ...params}``."""
    cfg = dict(config)
    name = cfg.pop("method", None)
    if name not in JUDGES:
        msg = f"unknown judge method {name!r}; choices: {sorted(JUDGES)}"
        raise ValueError(msg)
    return JUDGES[name](**cfg)


@dataclass
class JudgeStage(ProcessingStage[AudioTask, AudioTask]):
    """Attach a faithfulness verdict to every clip.

    Writes ``judge_faithful`` (bool), ``judge_heard`` (the transcripts), ``judge_reason`` and
    ``judge_method``. Follow it with :class:`FilterFaithfulStage` to drop unfaithful clips, or keep
    them for review.

    Args:
        judge: Judge config, see :func:`make_judge`.
        filepath_key: Key holding the clip path.
        number_key: Key holding the canonical target value.
    """

    judge: dict[str, Any] = field(default_factory=dict)
    filepath_key: str = "audio_filepath"
    number_key: str = "number"
    name: str = "judge"

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        self._judge = make_judge(self.judge)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.filepath_key, self.number_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["judge_faithful", "judge_heard", "judge_reason", "judge_method"]

    def process(self, task: AudioTask) -> AudioTask:
        t0 = time.perf_counter()
        verdict = self._judge.judge(task.data[self.filepath_key], str(task.data[self.number_key]))
        task.data["judge_faithful"] = bool(verdict["faithful"])
        task.data["judge_heard"] = verdict["heard"]
        task.data["judge_reason"] = verdict["reason"]
        task.data["judge_method"] = verdict["method"]
        self._log_metrics({"process_time": time.perf_counter() - t0, "faithful": float(verdict["faithful"])})
        return task


@dataclass
class FilterFaithfulStage(ProcessingStage[AudioTask, AudioTask]):
    """Drop clips whose ``key`` is false (the judge found the audio unfaithful)."""

    key: str = "judge_faithful"
    name: str = "filter_faithful"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.key]

    def process(self, task: AudioTask) -> AudioTask | None:
        return task if task.data.get(self.key) else None
