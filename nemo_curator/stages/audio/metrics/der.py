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

"""Diarization error rate (DER) against CHAT (.cha) reference annotations."""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, StageContract
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class DiarizationErrorRateStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """Score predicted diarization against a CHAT (``.cha``) transcript, as shipped with CallHome.

    The reference for ``<cha_dir>/<audio stem>.cha`` is read from its ``*X:\\t`` speaker lines and
    their ``\\x15start_end\\x15`` millisecond timestamps. DER is scored on 10 ms frames inside the
    reference span (UEM), skipping ``collar`` seconds around every reference boundary, after a greedy
    overlap-based speaker mapping. ``der_metrics_key`` gets ``der``, ``miss``, ``fa``, ``conf``, and
    ``correct`` (percent of reference speech), ``gt_speech_s``, ``pred_speech_s``, ``gt_speakers``,
    and ``pred_speakers``, or ``None`` when there is no reference file, no reference speech, or no
    prediction inside the UEM.

    Args:
        cha_dir: Directory holding one ``.cha`` file per recording.
        collar: Seconds ignored on each side of every reference segment boundary.
        audio_filepath_key: Field whose file stem names the ``.cha`` file.
        diar_segments_key: Field holding predicted ``{start, end, speaker}`` intervals.
        der_metrics_key: Output field for the metrics dictionary.
    """

    cha_dir: str
    collar: float = 0.25
    audio_filepath_key: str = "audio_filepath"
    diar_segments_key: str = "diar_segments"
    der_metrics_key: str = "der_metrics"
    name: str = "DiarizationErrorRate"

    INTERNAL_KEY_FIELDS = frozenset({"der_metrics_key"})

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, self.diar_segments_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.der_metrics_key]

    def describe(self) -> StageContract:
        return StageContract(
            reads=IOSpec(data_keys=[self.audio_filepath_key, self.diar_segments_key]),
            # Always written; None when there is no reference or no in-UEM prediction.
            writes=IOSpec(data_keys=[self.der_metrics_key]),
            gates=Gates(per_row_independent=True),
        )

    def process(self, task: AudioTask) -> AudioTask:
        cha_file = Path(self.cha_dir) / f"{Path(task.data[self.audio_filepath_key]).stem}.cha"
        metrics = None
        if cha_file.exists():
            gt, uem_start, uem_end = self._parse_cha(cha_file)
            metrics = self._compute_der(gt, task.data[self.diar_segments_key], uem_start, uem_end)
        task.data[self.der_metrics_key] = metrics
        return task

    @staticmethod
    def _parse_cha(path: Path) -> tuple[list[dict], float, float]:
        """Parse a CHA file → (segments, uem_start, uem_end)."""
        segs: list[dict] = []
        with open(path) as f:
            for line in f:
                m = re.match(r"^\*([A-Z]):\t", line)
                ts = re.search(r"\x15(\d+)_(\d+)\x15", line)
                if m and ts:
                    segs.append(
                        {"speaker": m.group(1), "start": int(ts.group(1)) / 1000, "end": int(ts.group(2)) / 1000}
                    )
        if not segs:
            return segs, 0.0, 0.0
        return segs, min(s["start"] for s in segs), max(s["end"] for s in segs)

    def _compute_der(  # noqa: C901, PLR0912
        self,
        gt: list[dict],
        pred: list[dict],
        uem_start: float,
        uem_end: float,
    ) -> dict | None:
        """Frame-level DER restricted to UEM region with collar tolerance."""
        if not gt or not pred:
            return None

        pred = [
            {"speaker": s["speaker"], "start": max(s["start"], uem_start), "end": min(s["end"], uem_end)}
            for s in pred
            if s["end"] > uem_start and s["start"] < uem_end
        ]
        pred = [s for s in pred if s["end"] > s["start"]]
        if not pred:
            return None

        collar_zones: list[tuple[float, float]] = []
        if self.collar > 0:
            for s in gt:
                collar_zones.append((s["start"] - self.collar, s["start"] + self.collar))
                collar_zones.append((s["end"] - self.collar, s["end"] + self.collar))

        def in_collar(t: float) -> bool:
            return any(lo <= t <= hi for lo, hi in collar_zones)

        # Greedy speaker mapping by overlap
        mv: Counter = Counter()
        for g in gt:
            for p in pred:
                ov = min(g["end"], p["end"]) - max(g["start"], p["start"])
                if ov > 0:
                    mv[(g["speaker"], p["speaker"])] += ov
        used, mapping = set(), {}
        for _v, gs, ps in sorted([(v, g, p) for (g, p), v in mv.items()], reverse=True):
            if gs not in mapping and ps not in used:
                mapping[gs] = ps
                used.add(ps)
        inv = {v: k for k, v in mapping.items()}

        # Frame-level scoring
        step = 0.01
        nf = int((uem_end - uem_start) / step) + 1
        gf: dict[int, set] = {}
        pf: dict[int, set] = {}
        for s in gt:
            for i in range(max(0, int((s["start"] - uem_start) / step)), min(nf, int((s["end"] - uem_start) / step))):
                gf.setdefault(i, set()).add(s["speaker"])
        for s in pred:
            mm = inv.get(s["speaker"], f"x_{s['speaker']}")
            for i in range(max(0, int((s["start"] - uem_start) / step)), min(nf, int((s["end"] - uem_start) / step))):
                pf.setdefault(i, set()).add(mm)

        miss = fa = conf = correct = total = 0
        for i in range(nf):
            t = uem_start + i * step
            if in_collar(t):
                continue
            gs = gf.get(i, set())
            ps = pf.get(i, set())
            if gs:
                total += len(gs)
                for s in gs:
                    if s in ps:
                        correct += 1
                    elif ps:
                        conf += 1
                    else:
                        miss += 1
            fa += len(ps - gs if gs else ps)

        ts = total * step
        if ts == 0:
            return None
        return {
            "der": (miss + fa + conf) * step / ts * 100,
            "miss": miss * step / ts * 100,
            "fa": fa * step / ts * 100,
            "conf": conf * step / ts * 100,
            "correct": correct * step / ts * 100,
            "gt_speech_s": ts,
            "pred_speech_s": sum(s["end"] - s["start"] for s in pred),
            "gt_speakers": len({s["speaker"] for s in gt}),
            "pred_speakers": len({s["speaker"] for s in pred}),
        }
