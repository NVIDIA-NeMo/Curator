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

"""Minimal Montreal Forced Aligner (MFA 3.x) runtime for the early cut-off resolver.

Writes a one-speaker corpus of candidate utterances, run
``mfa align`` once, then read the final word/phone from the long-format TextGrids and the per-utterance
likelihoods from MFA's SQLite working database (read-only).
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import subprocess
from collections import defaultdict
from statistics import median
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from pathlib import Path

JsonDict = dict[str, Any]

_ITEM_RE = re.compile(r"^\s*item\s*\[\d+\]\s*:\s*$", re.IGNORECASE)
_INTERVAL_RE = re.compile(r"^\s*intervals\s*\[\d+\]\s*:\s*$", re.IGNORECASE)
_PROPERTY_RE = re.compile(r"^\s*(name|class|xmin|xmax|text)\s*=\s*(.*?)\s*$", re.IGNORECASE)
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SILENCE = frozenset({"", "sil", "sp", "spn", "<eps>", "<unk>", "[noise]", "noise"})
_MAX_NAME_LEN = 96
_MAX_ERROR_TAIL = 4000


def safe_name(value: Any, fallback: str = "utterance") -> str:  # noqa: ANN401
    cleaned = _SAFE_RE.sub("_", str(value).strip()).strip("._-")
    return (cleaned or fallback)[:_MAX_NAME_LEN]


def write_corpus_utterance(corpus_dir: Path, key: str, waveform: Any, text: str, *, sample_rate: int) -> None:  # noqa: ANN401
    """Write one utterance of the single-speaker MFA corpus: ``candidate/<key>.wav`` (PCM16) + ``.lab``."""
    import soundfile as sf

    speaker = corpus_dir / "candidate"
    speaker.mkdir(parents=True, exist_ok=True)
    sf.write(speaker / f"{key}.wav", waveform, sample_rate, subtype="PCM_16")
    (speaker / f"{key}.lab").write_text(str(text).strip() + "\n", encoding="utf-8")


def build_align_command(  # noqa: PLR0913
    executable: str,
    corpus_dir: Path,
    dictionary: str,
    acoustic_model: str,
    output_dir: Path,
    temporary_dir: Path,
    *,
    num_jobs: int | None = None,
    clean: bool = True,
    extra_args: Sequence[str] = (),
    command_template: Sequence[str] | None = None,
) -> list[str]:
    """``mfa align`` argv; ``command_template`` tokens may use ``{executable} {corpus} {dictionary}
    {acoustic_model} {output} {temporary_directory}``."""
    values = {
        "executable": executable,
        "corpus": str(corpus_dir),
        "dictionary": dictionary,
        "acoustic_model": acoustic_model,
        "output": str(output_dir),
        "temporary_directory": str(temporary_dir),
    }
    if command_template:
        return [str(token).format(**values) for token in command_template]
    # MFA 3.x accepts these align options after the positional arguments.
    return [
        executable,
        "align",
        str(corpus_dir),
        dictionary,
        acoustic_model,
        str(output_dir),
        "--temporary_directory",
        str(temporary_dir),
        *(["--clean"] if clean else []),
        "--no_final_clean",
        *(["--num_jobs", str(num_jobs)] if num_jobs is not None else []),
        *map(str, extra_args),
    ]


def run_alignment(
    command: Sequence[str],
    *,
    runner: Callable[..., Any] = subprocess.run,
    timeout_sec: float | None = None,
    env: Mapping[str, str] | None = None,
) -> Any:  # noqa: ANN401
    """Run an MFA command (raising on failure or timeout) and return the completed process."""
    try:
        return runner(
            list(command),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=dict(env) if env is not None else None,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        msg = f"MFA exited with code {exc.returncode}: {detail[-_MAX_ERROR_TAIL:]}"
        raise RuntimeError(msg) from exc


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':  # noqa: PLR2004
        return value[1:-1].replace('""', '"')
    return value


def parse_long_textgrid(path: Path) -> list[JsonDict]:  # noqa: C901
    """Parse a long-format Praat TextGrid into interval tiers ``[{name, class, intervals: [{xmin, xmax, text}]}]``."""
    tiers: list[JsonDict] = []
    tier: JsonDict | None = None
    interval: JsonDict | None = None

    def finish_interval() -> None:
        nonlocal interval
        if tier is not None and interval is not None and {"xmin", "xmax", "text"} <= interval.keys():
            tier.setdefault("intervals", []).append(interval)
        interval = None

    def finish_tier() -> None:
        nonlocal tier
        finish_interval()
        if tier is not None and str(tier.get("class", "")).lower() == "intervaltier":
            tiers.append(tier)
        tier = None

    with path.open("r", encoding="utf-8-sig") as source:
        for line in source:
            if _ITEM_RE.match(line):
                finish_tier()
                tier = {"intervals": []}
                continue
            if tier is None:
                continue
            if _INTERVAL_RE.match(line):
                finish_interval()
                interval = {}
                continue
            match = _PROPERTY_RE.match(line)
            if not match:
                continue
            name, raw = match.groups()
            target = interval if interval is not None else tier
            target[name.lower()] = float(raw) if name.lower() in {"xmin", "xmax"} else _unquote(raw)
    finish_tier()
    if not tiers:
        msg = "MFA output contains no long TextGrid interval tiers"
        raise ValueError(msg)
    return tiers


def _tier_kind(name: Any) -> str | None:  # noqa: ANN401
    value = re.sub(r"[^a-z]", "", str(name).casefold())
    if "word" in value:
        return "word"
    if "phone" in value:
        return "phone"
    return None


def extract_textgrid_evidence(path: Path, *, audio_duration_sec: float, eof_threshold_sec: float) -> JsonDict:
    """Final non-silence word/phone of an MFA TextGrid and whether the last phone touches the audio end."""
    tiers = parse_long_textgrid(path)
    result: JsonDict = {"alignment_status": "ALIGNED", "textgrid_path": str(path)}
    for kind in ("word", "phone"):
        intervals = [
            item
            for tier in tiers
            if _tier_kind(tier.get("name")) == kind
            for item in tier.get("intervals", [])
            if str(item.get("text", "")).strip().casefold() not in _SILENCE
        ]
        if not intervals:
            msg = f"TextGrid has no non-silence {kind} interval"
            raise ValueError(msg)
        final = max(intervals, key=lambda item: (float(item["xmax"]), float(item["xmin"])))
        start, end = float(final["xmin"]), float(final["xmax"])
        result.update(
            {
                f"final_{kind}_label": str(final["text"]).strip(),
                f"final_{kind}_start_sec": start,
                f"final_{kind}_end_sec": end,
                f"final_{kind}_duration_sec": max(0.0, end - start),
            }
        )
    gap = audio_duration_sec - float(result["final_phone_end_sec"])
    result.update(
        {
            "audio_duration_sec": audio_duration_sec,
            "end_to_audio_eof_sec": gap,
            "touches_eof": abs(gap) <= eof_threshold_sec,
        }
    )
    return result


def discover_textgrids(root: Path) -> dict[str, Path]:
    """``{casefolded stem: path}`` of every TextGrid under ``root``; ambiguous stems are dropped."""
    result: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.casefold() == ".textgrid":
            key = path.stem.casefold()
            if key in result:
                duplicates.add(key)
            else:
                result[key] = path
    for key in duplicates:
        result.pop(key, None)
    return result


def discover_database(root: Path) -> Path | None:
    """The single ``*.db`` MFA database under ``root``, or None if there is not exactly one."""
    databases = sorted(path for path in root.rglob("*.db") if path.is_file())
    return databases[0] if len(databases) == 1 else None


def _finite_or_none(value: Any) -> float | None:  # noqa: ANN401
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None  # noqa: PLR0124


_LIKELIHOOD_QUERY = """
    SELECT f.name, u.alignment_log_likelihood, u.speech_log_likelihood,
           u.snr, u.duration_deviation, u.max_running_short_interval,
           p.phone, pi.begin, pi.end, pi.phone_goodness, pi.intensity
    FROM utterance AS u
    JOIN file AS f ON f.id = u.file_id
    LEFT JOIN phone_interval AS pi ON pi.utterance_id = u.id
    LEFT JOIN phone AS p ON p.id = pi.phone_id
    ORDER BY f.name, pi.end, pi.begin
"""


def sqlite_likelihood_evidence(database: Path, utterance_ids: Iterable[str]) -> dict[str, JsonDict]:
    """Per-utterance likelihood evidence from MFA's working database (opened read-only).

    Keys are casefolded utterance ids; utterances without phone intervals get ``likelihood_status=UNAVAILABLE``.
    """
    wanted = {str(value).casefold() for value in utterance_ids}
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    by_name: defaultdict[str, list[tuple[Any, ...]]] = defaultdict(list)
    summaries: dict[str, tuple[Any, ...]] = {}
    try:
        for record in connection.execute(_LIKELIHOOD_QUERY):
            name = str(record[0]).casefold()
            if name not in wanted:
                continue
            summaries[name] = tuple(record[1:6])
            if record[6] is not None and record[9] is not None:
                by_name[name].append(tuple(record[6:]))
    finally:
        connection.close()
    output: dict[str, JsonDict] = {}
    for name in wanted:
        summary, intervals = summaries.get(name), by_name.get(name, [])
        if summary is None or not intervals:
            output[name] = {"likelihood_status": "UNAVAILABLE"}
            continue
        final = max(intervals, key=lambda item: (float(item[2]), float(item[1])))
        previous = [float(item[3]) for item in intervals if item is not final][-3:]
        alignment = _finite_or_none(summary[0])
        speech = _finite_or_none(summary[1])
        final_score = float(final[3])
        output[name] = {
            "likelihood_status": "AVAILABLE",
            "alignment_log_likelihood_per_frame": alignment,
            "speech_log_likelihood": speech,
            "snr": _finite_or_none(summary[2]),
            "duration_deviation": _finite_or_none(summary[3]),
            "max_running_short_interval": _finite_or_none(summary[4]),
            "final_phone_path_loglike": final_score,
            "final_phone_intensity": _finite_or_none(final[4]),
            "final_phone_db_label": str(final[0]),
            "final_phone_db_start_sec": float(final[1]),
            "final_phone_db_end_sec": float(final[2]),
            "final_vs_alignment_loglike": final_score - alignment if alignment is not None else None,
            "final_vs_speech_loglike": final_score - speech if speech is not None else None,
            "final_vs_local_loglike": final_score - median(previous) if previous else None,
        }
    return output


def preserve_database(database: Path | None, destination: Path | None) -> Path | None:
    """Copy the MFA database to ``destination`` when both are given; return the kept path."""
    if database is None or destination is None:
        return database
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(database, destination)
    return destination
