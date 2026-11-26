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

"""Pure cut planning for long-form recordings (numpy only, no model, no Curator imports).

Given per-frame voice-activity probabilities and diarization intervals for one
recording, :func:`plan_longform_cuts` returns chunks that TILE the recording:
``[0, c1), [c1, c2), ..., [ck, duration)``. Nothing is dropped or trimmed, so
transcripts of the speech chunks, joined in order, describe the original audio.

Rules
  * Every speech chunk is at most ``max_sec`` long (a hard model-input limit,
    e.g. the 300 s Qwen3-Omni feature extractor) and, where the audio allows,
    at least ``min_sec``; lengths are pulled towards ``target_sec``.
  * Cuts go where nobody speaks: long VAD pauses score best, the diarizer must
    also be silent there, and a speaker hand-off is a small bonus only.
    A cut inside speech is used only when nothing else fits the length window.
  * Stretches of at least ``skip_min_sec`` where the diarizer hears nobody and
    VAD hears speech in at most ``skip_max_vad_speech_frac`` of the frames become
    ``silence`` chunks: they are kept in the tiling but need no ASR. A VAD veto
    is required because the diarizer misses chanting and singing.
  * Speech fragments shorter than ``merge_below_sec`` (left between two long
    silences) are merged with a neighbour, silence included, if the result still
    fits ``max_sec``.

Chosen and validated on 30 long-form granary-v2 recordings (16.1 h, en/es/pt)
with Silero VAD and Nemotron-3-Diarization: no cut splits a word (Parakeet
timestamp check), 62% of inner cuts fall on a sentence end, and the skipped
silence (8.4 min) contained no reference-transcript content.

Algorithm: candidate cut points (pause midpoints, speaker hand-offs centred in
their pause, and a forced quietest frame per 2 s as a fallback) are scored, then
a dynamic programme over candidates minimises
``sum_chunks [C + lam * ((L - T) / T)^2] - sum_cuts score`` with
``min_sec <= L <= max_sec``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

SILENCE = "silence"
SPEECH = "speech"

_SPEECH_ENTRY_LEN = 2  # [start, end]
_SILENCE_ENTRY_LEN = 3  # [start, end, "silence"]
_EPS_SEC = 1e-6
_MIN_CHUNK_SEC = 1e-3  # stretches shorter than this are rounding residue, not chunks
_CLEAN_HANDOFF_OVERLAP_SEC = 0.1  # speaker change with at most this much overlap counts as clean


@dataclass(frozen=True)
class LongformCutConfig:
    """Planner settings. Defaults are the validated production values."""

    target_sec: float = 240.0
    min_sec: float = 120.0
    max_sec: float = 290.0
    chunk_cost: float = 20.0
    length_weight: float = 30.0
    # VAD hysteresis and minimum pause length
    vad_onset: float = 0.5
    vad_offset: float = 0.35
    min_pause_sec: float = 0.15
    # boundary scoring: (min pause seconds, points) tiers, longest matching wins
    pause_tiers: tuple[tuple[float, float], ...] = ((1.0, 8.0), (0.6, 5.0), (0.3, 2.0), (1e-9, 1.0))
    pause_weight: float = 1.0
    diar_silent_bonus: float = 4.0
    diar_speech_penalty: float = 10.0
    vad_speech_penalty: float = 10.0
    speaker_change_bonus: float = 2.0
    overlap_change_penalty: float = 2.0
    forced_cut_penalty: float = 50.0
    forced_cut_step_sec: float = 2.0
    # silence skipping
    skip_min_sec: float = 10.0
    skip_margin_sec: float = 0.5
    skip_max_vad_speech_frac: float = 0.10
    # short fragment merge (0 disables)
    merge_below_sec: float = 30.0

    def validate(self) -> None:
        if not 0 < self.min_sec <= self.target_sec <= self.max_sec:
            msg = f"need 0 < min_sec <= target_sec <= max_sec, got {self.min_sec}, {self.target_sec}, {self.max_sec}"
            raise ValueError(msg)
        if not 0.0 <= self.vad_offset <= self.vad_onset <= 1.0:
            msg = f"need 0 <= vad_offset <= vad_onset <= 1, got {self.vad_offset}, {self.vad_onset}"
            raise ValueError(msg)
        if self.forced_cut_step_sec <= 0 or self.forced_cut_step_sec * 2 > self.min_sec:
            msg = "forced_cut_step_sec must be > 0 and at most min_sec / 2 so a feasible plan always exists"
            raise ValueError(msg)
        if self.skip_min_sec <= 2 * self.skip_margin_sec:
            msg = "skip_min_sec must exceed twice skip_margin_sec"
            raise ValueError(msg)


@dataclass
class Chunk:
    """One element of the tiling, in seconds of the original recording."""

    start: float
    end: float
    kind: str  # SPEECH or SILENCE
    end_cut: dict | None = None  # how the boundary at ``end`` was chosen (diagnostics only)
    merged: int = 1

    @property
    def duration(self) -> float:
        return self.end - self.start


def _speech_mask(prob: np.ndarray, onset: float, offset: float) -> np.ndarray:
    speech = np.zeros(len(prob), bool)
    state = False
    for i, p in enumerate(prob):
        state = p >= onset if not state else p >= offset
        speech[i] = state
    return speech


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[i, j) index runs where ``mask`` is True."""
    if len(mask) == 0:
        return []
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


class _Planner:
    def __init__(
        self, prob: np.ndarray, frame_sec: float, diar: list[tuple[float, float, str]], cfg: LongformCutConfig
    ):
        self.prob = np.asarray(prob, dtype=np.float64)
        self.fr = float(frame_sec)
        self.diar = sorted(diar, key=lambda x: x[0])
        self.cfg = cfg
        n = len(self.prob)
        speech = _speech_mask(self.prob, cfg.vad_onset, cfg.vad_offset)
        self.pauses = [(a * self.fr, b * self.fr) for a, b in _runs(~speech) if (b - a) * self.fr >= cfg.min_pause_sec]
        self._pause_starts = np.array([a for a, _ in self.pauses])
        self._pause_ends = np.array([b for _, b in self.pauses])
        act = np.zeros(n, np.int32)
        for a, b, _ in self.diar:
            act[int(a / self.fr) : int(b / self.fr) + 1] += 1
        self.act = act

    # ---- lookups ---------------------------------------------------------------------------
    def pause_at(self, t: float) -> tuple[float, float] | None:
        k = int(np.searchsorted(self._pause_starts, t, "right")) - 1
        if k >= 0 and t <= self._pause_ends[k]:
            return self.pauses[k]
        return None

    def pause_len_at(self, t: float) -> float:
        iv = self.pause_at(t)
        return iv[1] - iv[0] if iv else 0.0

    # ---- candidates --------------------------------------------------------------------------
    def score_at(self, t: float, spk_kind: str | None) -> tuple[float, float, bool]:
        cfg = self.cfg
        i = min(len(self.prob) - 1, int(t / self.fr))
        g = self.pause_len_at(t)
        sc = 0.0
        for thr, pts in cfg.pause_tiers:
            if g >= thr:
                sc = pts
                break
        sc += cfg.pause_weight * min(g, 3.0)
        lo, hi = max(0, i - 3), min(len(self.act), i + 4)  # +-0.1 s tolerance on diarizer edges
        diar_silent = bool(self.act[lo:hi].max() == 0)
        sc += cfg.diar_silent_bonus if diar_silent else -cfg.diar_speech_penalty
        if self.prob[i] >= cfg.vad_onset:
            sc -= cfg.vad_speech_penalty
        if spk_kind == "spk":
            sc += cfg.speaker_change_bonus
        elif spk_kind == "spk_overlap":
            sc -= cfg.overlap_change_penalty
        return sc, g, diar_silent

    def speaker_changes(self) -> list[tuple[float, float]]:
        """Hand-offs between turns of DIFFERENT speakers; backchannels inside a turn are ignored."""
        out = []
        cover_end, cover_spk = -1.0, None
        for a, b, spk in self.diar:
            if cover_spk is not None and spk != cover_spk and a >= cover_end - 0.1 and b > cover_end:
                lo, hi = min(cover_end, a) - 0.2, max(cover_end, a) + 0.2
                i0, i1 = max(0, int(lo / self.fr)), min(len(self.prob), int(hi / self.fr) + 1)
                if i1 > i0:
                    t = (i0 + int(np.argmin(self.prob[i0:i1])) + 0.5) * self.fr
                    out.append((t, max(0.0, cover_end - a)))
            if b > cover_end:
                cover_end, cover_spk = b, spk
        return out

    def candidates(self, duration: float) -> list[dict]:
        cfg = self.cfg
        c: dict[float, dict] = {}

        def add(t: float, kind: str, forced: bool = False, **kw) -> None:
            t = round(float(t), 3)
            if not 0 < t < duration:
                return
            sc, g, ds = self.score_at(t, kind if kind.startswith("spk") else None)
            if forced:
                sc -= cfg.forced_cut_penalty
            if t not in c or c[t]["score"] < sc:
                c[t] = {"t": t, "kind": kind, "score": sc, "gap": round(g, 3), "diar_silent": ds, **kw}

        for a, b in self.pauses:
            add((a + b) / 2, "pause")
        for t, ov in self.speaker_changes():
            iv = self.pause_at(t)  # centre the cut in the pause around the hand-off
            cut = (iv[0] + iv[1]) / 2 if iv else t
            add(cut, "spk" if ov <= _CLEAN_HANDOFF_OVERLAP_SEC else "spk_overlap", overlap=round(ov, 3))
        step = max(1, int(cfg.forced_cut_step_sec / self.fr))
        for i in range(0, len(self.prob), step):
            j = i + int(np.argmin(self.prob[i : i + step]))
            add((j + 0.5) * self.fr, "forced", forced=True)
        return sorted(c.values(), key=lambda x: x["t"])

    # ---- silence skipping ---------------------------------------------------------------------
    def skippable_silences(self) -> list[tuple[float, float]]:
        cfg = self.cfg
        n = len(self.prob)
        out = []
        for i, j in _runs(self.act == 0):
            if (j - i) * self.fr < cfg.skip_min_sec:
                continue
            if np.mean(self.prob[i:j] >= cfg.vad_onset) > cfg.skip_max_vad_speech_frac:
                continue  # VAD veto: chanting / singing / quiet speech the diarizer missed
            a = 0.0 if i == 0 else i * self.fr + cfg.skip_margin_sec
            b = j * self.fr if j == n else j * self.fr - cfg.skip_margin_sec
            out.append((a, b))
        return out

    # ---- DP ----------------------------------------------------------------------------------------
    def plan_stretch(self, cands: list[dict], s0: float, s1: float) -> list[dict]:
        cfg = self.cfg
        if s1 - s0 <= cfg.max_sec:
            return []
        target = cfg.target_sec
        inner = [c for c in cands if s0 < c["t"] < s1]
        nodes = [{"t": s0, "score": 0.0, "kind": "start"}, *inner, {"t": s1, "score": 0.0, "kind": "end"}]
        ts = np.array([x["t"] for x in nodes])
        best = np.full(len(nodes), np.inf)
        best[0] = 0.0
        back = np.full(len(nodes), -1)
        for j in range(1, len(nodes)):
            lo = np.searchsorted(ts, ts[j] - cfg.max_sec, "left")
            hi = np.searchsorted(ts, ts[j] - cfg.min_sec, "right")
            for i in range(lo, min(hi, j)):
                if best[i] == np.inf:
                    continue
                length = ts[j] - ts[i]
                v = (
                    best[i]
                    + cfg.chunk_cost
                    + cfg.length_weight * ((length - target) / target) ** 2
                    - nodes[j]["score"]
                )
                if v < best[j]:
                    best[j], back[j] = v, i
        j = len(nodes) - 1
        if best[j] == np.inf:
            msg = f"no feasible cut plan for stretch {s0:.1f}-{s1:.1f} s"
            raise RuntimeError(msg)
        cuts = []
        while back[j] > 0:
            j = back[j]
            cuts.append(nodes[j])
        return cuts[::-1]


def plan_longform_cuts(
    prob: np.ndarray,
    frame_sec: float,
    diar: list[tuple[float, float, str]],
    duration: float,
    cfg: LongformCutConfig | None = None,
) -> list[Chunk]:
    """Plan chunks tiling ``[0, duration]``.

    Args:
        prob: speech probability per VAD frame (frame ``i`` covers ``[i, i+1) * frame_sec``).
        frame_sec: VAD frame hop in seconds (Silero at 16 kHz: 512 / 16000 = 0.032).
        diar: diarization intervals ``(start_sec, end_sec, speaker_label)``.
        duration: recording length in seconds.
        cfg: planner settings; defaults are the validated production values.
    """
    cfg = cfg or LongformCutConfig()
    cfg.validate()
    p = _Planner(prob, frame_sec, diar, cfg)
    silences = p.skippable_silences()
    cands = p.candidates(duration)

    stretches, pos = [], 0.0
    for a, b in silences:
        stretches.append((SPEECH, pos, a))
        stretches.append((SILENCE, a, b))
        pos = b
    stretches.append((SPEECH, pos, duration))

    chunks: list[Chunk] = []
    for kind, s0, s1 in stretches:
        if s1 - s0 <= _MIN_CHUNK_SEC:
            continue
        if kind == SILENCE:
            chunks.append(Chunk(s0, s1, SILENCE, {"kind": "silence_edge"}))
            continue
        cuts = p.plan_stretch(cands, s0, s1)
        bounds = [s0, *(c["t"] for c in cuts), s1]
        for k in range(len(bounds) - 1):
            ec = cuts[k] if k < len(cuts) else {"kind": "silence_edge" if s1 < duration else "end"}
            chunks.append(Chunk(bounds[k], bounds[k + 1], SPEECH, ec))

    if cfg.merge_below_sec > 0:
        _merge_short_fragments(chunks, cfg)
    return chunks


def _merge_short_fragments(chunks: list[Chunk], cfg: LongformCutConfig) -> None:
    """Merge speech fragments shorter than ``merge_below_sec`` into a neighbour (in place).

    Options are: the fragment plus the silence and speech chunk on either side, or plus
    only the adjacent chunk; the shortest group that still fits ``max_sec`` wins. The
    absorbed silence simply goes to ASR, nothing is lost.
    """
    changed = True
    while changed:
        changed = False
        for i, ch in enumerate(chunks):
            if ch.kind != SPEECH or ch.duration >= cfg.merge_below_sec:
                continue
            opts = []
            for lo, hi in ((i - 2, i), (i, i + 2), (i - 1, i), (i, i + 1)):
                if lo < 0 or hi >= len(chunks):
                    continue
                grp = chunks[lo : hi + 1]
                if any(g.kind == SPEECH for g in grp if g is not ch):
                    span = grp[-1].end - grp[0].start
                    if span <= cfg.max_sec:
                        opts.append((span, lo, hi))
            if opts:
                _, lo, hi = min(opts)
                chunks[lo : hi + 1] = [
                    Chunk(chunks[lo].start, chunks[hi].end, SPEECH, chunks[hi].end_cut, merged=hi - lo + 1)
                ]
                changed = True
                break


def chunks_to_cuts(chunks: list[Chunk], ndigits: int = 3) -> list[list]:
    """Compact, JSON-safe form: ``[start, end]`` per speech chunk, ``[start, end, "silence"]`` otherwise."""
    out: list[list] = []
    for ch in chunks:
        row: list = [round(ch.start, ndigits), round(ch.end, ndigits)]
        if ch.kind == SILENCE:
            row.append(SILENCE)
        out.append(row)
    return out


def cuts_to_spans(cuts: list[list]) -> list[tuple[float, float, bool]]:
    """Parse the compact form back to ``(start, end, is_speech)``."""
    out = []
    for row in cuts:
        if len(row) not in (_SPEECH_ENTRY_LEN, _SILENCE_ENTRY_LEN) or (
            len(row) == _SILENCE_ENTRY_LEN and row[2] != SILENCE
        ):
            msg = f"invalid cut entry {row!r}; expected [start, end] or [start, end, {SILENCE!r}]"
            raise ValueError(msg)
        out.append((float(row[0]), float(row[1]), len(row) == _SPEECH_ENTRY_LEN))
    return out


def check_tiling(cuts: list[list], duration: float, max_sec: float, tol: float = 1e-2) -> str | None:  # noqa: PLR0911
    """Return an error message if ``cuts`` do not tile ``[0, duration]`` within limits, else None."""
    spans = cuts_to_spans(cuts)
    if not spans:
        return "empty cut plan"
    if abs(spans[0][0]) > _EPS_SEC:
        return f"first chunk starts at {spans[0][0]}"
    if abs(spans[-1][1] - duration) > tol:
        return f"last chunk ends at {spans[-1][1]}, duration {duration}"
    for (_, e, _), (s, _, _) in itertools.pairwise(spans):
        if abs(e - s) > _EPS_SEC:
            return f"gap or overlap at {e} / {s}"
    for s, e, is_speech in spans:
        if e <= s:
            return f"empty chunk {s}-{e}"
        if is_speech and e - s > max_sec + _EPS_SEC:
            return f"speech chunk {s}-{e} longer than {max_sec} s"
    return None
