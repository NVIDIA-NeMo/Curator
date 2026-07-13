"""Shared helpers for the David AI MFA pipeline."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

MFA_ROOT_DIR_DEFAULT = "~/MFA_models"

POSTPROCESSED_RE = re.compile(r"^(.+)_postprocessed\.wav$")
SILENCE_TOKENS = {"", "sil", "sp", "spn", "<eps>"}

logger = logging.getLogger(__name__)

# Hard wall-clock cap for any single ffmpeg subprocess. Without this, a wedged
# ffmpeg (e.g. an internal futex deadlock seen when many run in a worker pool)
# blocks its caller forever and hangs the whole shard. On timeout, subprocess
# kills the child and raises TimeoutExpired, which callers treat as a failure.
# Configurable via FFMPEG_TIMEOUT_S: the final multi-speaker `amix` of long
# sessions can exceed the default when all node CPUs are saturated (mix reruns
# use a higher value and lower concurrency).
def _ffmpeg_timeout_s() -> int:
    raw = os.environ.get("FFMPEG_TIMEOUT_S", "").strip()
    if raw:
        try:
            value = int(float(raw))
            if value > 0:
                return value
        except ValueError:
            pass
    return 600


FFMPEG_TIMEOUT_S = _ffmpeg_timeout_s()


def ffmpeg_executable() -> str:
    """Return ffmpeg binary path; honors FFMPEG_BIN for cluster static builds."""
    env_bin = os.environ.get("FFMPEG_BIN", "").strip()
    if env_bin:
        return env_bin
    return shutil.which("ffmpeg") or "ffmpeg"


T = TypeVar("T")
R = TypeVar("R")


class PipelineError(Exception):
    """Raised when a pipeline stage cannot continue."""


def mfa_models_root() -> Path:
    return Path(os.environ.get("MFA_ROOT_DIR", MFA_ROOT_DIR_DEFAULT)).expanduser().resolve()


def resolve_mfa_dict(mfa_dict: str) -> Path:
    for candidate in (
        mfa_models_root() / "pretrained_models" / "dictionary" / f"{mfa_dict}.dict",
        mfa_models_root() / "pretrained_models" / "dictionary" / f"{mfa_dict}.txt",
        Path(mfa_dict).expanduser(),
    ):
        if candidate.is_file():
            return candidate.resolve()
    msg = f"MFA dictionary not found for {mfa_dict!r}"
    raise FileNotFoundError(msg)


def resolve_mfa_acoustic_model(mfa_acoustic: str) -> str:
    """Resolve an acoustic model name to its pretrained zip path.

    Passing the full zip path (instead of the bare model name) lets each MFA
    invocation use its own ``--temporary_directory`` without breaking model
    lookup, which is required for safe parallel execution.
    """
    direct = Path(mfa_acoustic).expanduser()
    if direct.is_file() or direct.is_dir():
        return str(direct.resolve())
    for candidate in (
        mfa_models_root() / "pretrained_models" / "acoustic" / f"{mfa_acoustic}.zip",
        mfa_models_root() / "pretrained_models" / "acoustic" / mfa_acoustic,
    ):
        if candidate.is_file() or candidate.is_dir():
            return str(candidate.resolve())
    return mfa_acoustic


def partition_list(items: list[T], num_parts: int) -> list[list[T]]:
    if num_parts <= 1 or not items:
        return [items]
    parts: list[list[T]] = [[] for _ in range(num_parts)]
    for i, item in enumerate(items):
        parts[i % num_parts].append(item)
    return [part for part in parts if part]


def append_mfa_g2p_args(cmd: list[str], *, g2p_path: str | Path | None) -> None:
    if g2p_path:
        cmd.extend(["--g2p_model_path", str(g2p_path)])


def mfa_subprocess_env(
    *,
    temp_root: Path,
    mfa_root: Path,
) -> dict[str, str]:
    """Build env for ``mfa`` subprocesses.

    Inside the pyxis container, ``PYTHONPATH`` includes ``/opt/venv/...`` which
    shadows the packed conda ``pynini``/OpenFST with a different wheel build.
    Mixed imports (conda ``montreal_forced_aligner`` + container ``pynini``)
    cause ``FstIOError: Read failed`` on G2P ``model.fst``.
    """
    env = os.environ.copy()
    env["TMPDIR"] = str(temp_root.parent)
    env["MFA_ROOT_DIR"] = str(mfa_root)

    conda_lib: Path | None = None
    mfa_env_dir = os.environ.get("MFA_ENV", "").strip()
    if mfa_env_dir:
        candidate = Path(mfa_env_dir) / "lib"
        if candidate.is_dir():
            conda_lib = candidate
    if conda_lib is None:
        mfa_bin = shutil.which("mfa")
        if mfa_bin:
            candidate = Path(mfa_bin).resolve().parent.parent / "lib"
            if candidate.is_dir():
                conda_lib = candidate
    if conda_lib is not None:
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{conda_lib}:{prev}" if prev else str(conda_lib)

    pp = env.get("PYTHONPATH", "")
    if pp:
        kept = [
            p
            for p in pp.split(os.pathsep)
            if p and "/opt/venv" not in p and "/opt/Export-Deploy" not in p
        ]
        if kept:
            env["PYTHONPATH"] = os.pathsep.join(kept)
        else:
            env.pop("PYTHONPATH", None)
    return env


def _worker_g2p_arg(models_dir: Path, mfa_g2p: str) -> str | None:
    g2p_src = resolve_mfa_g2p_model(mfa_g2p)
    if g2p_src.is_dir():
        local_dir = models_dir / "g2p" / g2p_src.name
        if not (local_dir / "model.fst").is_file():
            local_dir.parent.mkdir(parents=True, exist_ok=True)
            if local_dir.exists():
                shutil.rmtree(local_dir)
            shutil.copytree(g2p_src, local_dir)
        return str(local_dir)
    if g2p_src.is_file():
        local = models_dir / g2p_src.name
        if not local.is_file():
            shutil.copy2(g2p_src, local)
        return str(local)
    return None


def setup_mfa_worker_root(
    worker_dir: Path,
    *,
    mfa_dict: Path,
    mfa_acoustic: str,
    mfa_g2p: str | None = None,
    source_mfa_root: Path | None = None,
) -> tuple[Path, Path, str, str | None]:
    """Prepare an isolated MFA root with local copies of lexicon and acoustic model.

    Returns (mfa_root, local_dict_path, acoustic_model_arg, g2p_model_arg) for ``mfa align``.
    """
    worker_dir = worker_dir.resolve()
    if worker_dir.exists():
        shutil.rmtree(worker_dir, ignore_errors=True)
    mfa_root = worker_dir / "mfa_root"
    models_dir = worker_dir / "models"

    models_dir.mkdir(parents=True, exist_ok=True)
    mfa_root.mkdir(parents=True, exist_ok=True)

    local_dict = models_dir / mfa_dict.name
    shutil.copy2(mfa_dict, local_dict)

    acoustic_src = Path(resolve_mfa_acoustic_model(mfa_acoustic))
    if acoustic_src.is_file() and acoustic_src.suffix == ".zip":
        local_zip = models_dir / acoustic_src.name
        shutil.copy2(acoustic_src, local_zip)
        acoustic_arg = str(local_zip)
        _extract_acoustic_zip(local_zip, mfa_root, source_mfa_root=source_mfa_root)
    elif acoustic_src.is_dir():
        local_acoustic = models_dir / "acoustic" / acoustic_src.name
        local_acoustic.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(acoustic_src, local_acoustic)
        acoustic_arg = str(local_acoustic)
    else:
        acoustic_arg = resolve_mfa_acoustic_model(mfa_acoustic)

    g2p_arg = _worker_g2p_arg(models_dir, mfa_g2p) if mfa_g2p else None

    global_config = mfa_root / "global_config.yaml"
    global_config.write_text(
        "\n".join(
            [
                "auto_server: true",
                "blas_num_threads: 1",
                "clean: false",
                "cleanup_textgrids: true",
                "database_limited_mode: false",
                "debug: false",
                "num_jobs: 3",
                "overwrite: false",
                "quiet: false",
                "seed: 0",
                "single_speaker: false",
                f"temporary_directory: {mfa_root}",
                "use_mp: true",
                "use_postgres: false",
                "use_threading: true",
                "verbose: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cmd_hist = mfa_root / "command_history.yaml"
    if cmd_hist.exists() or cmd_hist.is_symlink():
        cmd_hist.unlink(missing_ok=True)
    cmd_hist.symlink_to("/dev/null")

    return mfa_root, local_dict, acoustic_arg, g2p_arg


def _extract_acoustic_zip(
    zip_path: Path,
    mfa_root: Path,
    *,
    source_mfa_root: Path | None = None,
) -> None:
    import zipfile

    extracted_root = mfa_root / "extracted_models" / "acoustic"
    if source_mfa_root is not None:
        src_acoustic = source_mfa_root / "extracted_models" / "acoustic"
        if src_acoustic.is_dir():
            for src_dir in src_acoustic.iterdir():
                if not src_dir.is_dir():
                    continue
                dst_dir = extracted_root / src_dir.name
                if (dst_dir / "final.mdl").is_file():
                    continue
                shutil.copytree(src_dir, dst_dir)
                return

    extracted_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extracted_root)

    for _path in extracted_root.rglob("final.mdl"):
        return
    msg = f"acoustic zip did not contain final.mdl: {zip_path}"
    raise PipelineError(msg)


def resolve_mfa_g2p_model(mfa_g2p: str) -> Path:
    direct = Path(mfa_g2p).expanduser()
    if direct.is_file() or direct.is_dir():
        return direct.resolve()
    root = mfa_models_root()
    for candidate in (
        root / "extracted_models" / "g2p" / f"{mfa_g2p}_g2p",
        root / "extracted_models" / "g2p" / mfa_g2p,
        root / "pretrained_models" / "g2p" / mfa_g2p,
        root / "pretrained_models" / "g2p" / f"{mfa_g2p}.zip",
    ):
        if candidate.is_file() or candidate.is_dir():
            return candidate.resolve()
    msg = (
        f"MFA G2P model not found for {mfa_g2p!r} under "
        f"{root / 'pretrained_models' / 'g2p'} or {root / 'extracted_models' / 'g2p'}"
    )
    raise FileNotFoundError(
        msg
    )


def log_exception(context: str, exc: BaseException) -> None:
    logger.error("%s: %s", context, exc)
    logger.debug(traceback.format_exc())


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        with path.open(encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    msg = f"{path}:{line_no}: invalid JSON: {exc}"
                    raise ValueError(msg) from exc
    except OSError as exc:
        msg = f"cannot read {path}: {exc}"
        raise PipelineError(msg) from exc
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        msg = f"cannot write {path}: {exc}"
        raise PipelineError(msg) from exc


def append_jsonl(path: Path, row: dict, *, lock: threading.Lock | None = None) -> None:
    def _write() -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            msg = f"cannot append to {path}: {exc}"
            raise PipelineError(msg) from exc

    if lock is not None:
        with lock:
            _write()
    else:
        _write()


def thread_temp_root(base: Path) -> Path:
    """Per-thread scratch directory under *base* for parallel workers."""
    root = base / f"thread_{threading.get_ident()}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_thread_pool(
    items: list[T],
    fn: Callable[[T], R],
    *,
    workers: int = 1,
) -> list[R]:
    if workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


def run_process_pool(
    items: list[T],
    fn: Callable[[T], R],
    *,
    workers: int = 1,
) -> list[R]:
    if workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


def ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        msg = f"ffprobe not available for {path}: {exc}"
        raise RuntimeError(msg) from exc
    if result.returncode != 0 or not result.stdout.strip():
        msg = f"ffprobe failed for {path}: {result.stderr[-300:]}"
        raise RuntimeError(msg)
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        msg = f"ffprobe returned non-numeric duration for {path}"
        raise RuntimeError(msg) from exc


def extract_segment_wav(
    src: Path,
    dst: Path,
    start: float,
    end: float,
    *,
    padding: float = 0.0,
    max_duration: float | None = None,
) -> float | None:
    """Extract audio for MFA, optionally padded before/after the manifest interval.

    Returns the absolute extract start time in *src*, or None on failure.
    The clip spans [max(0, start-padding), min(max_duration, end+padding)] when
    *max_duration* is set, otherwise [max(0, start-padding), end+padding].
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    extract_start = max(0.0, start - padding)
    extract_end = end + padding
    if max_duration is not None:
        extract_end = min(max_duration, extract_end)
    duration = max(extract_end - extract_start, 0.01)
    cmd = [
        ffmpeg_executable(),
        "-nostdin",
        "-y",
        "-ss",
        f"{extract_start:.6f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.6f}",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        str(dst),
    ]
    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=FFMPEG_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffmpeg segment extract failed to start for %s: %s", src, exc)
        return None
    if result.returncode != 0:
        logger.warning(
            "ffmpeg segment extract failed (%s [%.3f-%.3f] pad=%.3f): %s",
            src,
            extract_start,
            extract_end,
            padding,
            result.stderr[-300:],
        )
        return None
    return extract_start


def map_segment_words_to_recording(
    words: list[tuple[float, float, str]],
    *,
    extract_start: float,
    extract_end: float,
) -> list[tuple[float, float, str]]:
    """Map MFA word times from a padded segment clip back to recording time.

    All MFA word intervals are kept as aligned (only bounded by the clip MFA ran on).
    """
    mapped: list[tuple[float, float, str]] = []
    for start, end, word in words:
        abs_start = start + extract_start
        abs_end = end + extract_start
        if abs_end <= extract_start or abs_start >= extract_end:
            continue
        mapped.append((abs_start, abs_end, word))
    return mapped


def parse_textgrid_words(tg_path: Path) -> list[tuple[float, float, str]]:
    import textgrid

    try:
        tg = textgrid.TextGrid.fromFile(str(tg_path))
        tier = tg.getFirst("words")
    except Exception as exc:
        msg = f"failed to parse TextGrid {tg_path}: {exc}"
        raise ValueError(msg) from exc

    words: list[tuple[float, float, str]] = []
    for iv in tier.intervals:
        mark = (iv.mark or "").strip()
        if mark and mark not in SILENCE_TOKENS:
            words.append((iv.minTime, iv.maxTime, mark))
    return words


def safe_parse_textgrid_words(tg_path: Path) -> list[tuple[float, float, str]]:
    try:
        return parse_textgrid_words(tg_path)
    except ImportError as exc:
        msg = (
            "textgrid package is required to parse MFA TextGrids "
            f"(pip install textgrid): {exc}"
        )
        raise PipelineError(
            msg
        ) from exc
    except Exception as exc:
        log_exception(f"TextGrid parse failed for {tg_path}", exc)
        return []


def write_textgrid(
    words: list[tuple[float, float, str]],
    output_path: Path,
    *,
    xmin: float = 0.0,
    xmax: float | None = None,
) -> None:
    if xmax is None:
        xmax = words[-1][1] + 0.01 if words else xmin + 0.01

    intervals: list[tuple[float, float, str]] = []
    prev_end = xmin
    for start, end, word in sorted(words, key=lambda x: x[0]):
        if start > prev_end + 0.001:
            intervals.append((prev_end, start, ""))
        intervals.append((start, end, word))
        prev_end = end
    if prev_end < xmax:
        intervals.append((prev_end, xmax, ""))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write('File type = "ooTextFile"\n')
        f.write('Object class = "TextGrid"\n\n')
        f.write(f"xmin = {xmin}\n")
        f.write(f"xmax = {xmax}\n")
        f.write("tiers? <exists>\n")
        f.write("size = 1\n")
        f.write("item []:\n")
        f.write("    item [1]:\n")
        f.write('        class = "IntervalTier"\n')
        f.write('        name = "words"\n')
        f.write(f"        xmin = {xmin}\n")
        f.write(f"        xmax = {xmax}\n")
        f.write(f"        intervals: size = {len(intervals)}\n")
        for i, (s, e, text) in enumerate(intervals, 1):
            safe = text.replace('"', '""')
            f.write(f"        intervals [{i}]:\n")
            f.write(f"            xmin = {s}\n")
            f.write(f"            xmax = {e}\n")
            f.write(f'            text = "{safe}"\n')


def merge_speech_intervals(
    intervals: list[tuple[float, float]],
    merge_gap: float,
) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and (start - merged[-1][1]) <= merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def merge_tagged_speech_intervals(
    intervals: list[tuple[float, float, str]],
    merge_gap: float,
) -> list[tuple[float, float, str]]:
    merged: list[tuple[float, float, str]] = []
    for start, end, label in sorted(intervals):
        if end <= start:
            continue
        if merged and (start - merged[-1][1]) <= merge_gap:
            prev_start, prev_end, prev_label = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end), prev_label)
        else:
            merged.append((start, end, label))
    return merged


def textgrid_to_rttm_lines(
    tg_path: Path,
    *,
    speaker: str,
    merge_gap: float = 0.2,
) -> list[str]:
    file_id = tg_path.stem
    try:
        intervals = [
            (start, end)
            for start, end, _ in parse_textgrid_words(tg_path)
        ]
    except Exception as exc:
        log_exception(f"RTTM conversion failed for {tg_path}", exc)
        return []

    merged = merge_speech_intervals(intervals, merge_gap)

    lines = []
    for start, end in merged:
        dur = end - start
        lines.append(
            f"SPEAKER {file_id} 1 {start:.6f} {dur:.6f} <NA> <NA> {speaker} <NA> <NA>"
        )
    return lines


def discover_sessions(data_root: Path) -> list[Path]:
    sessions: list[Path] = []
    for path in data_root.iterdir():
        if path.is_symlink():
            # Cluster data_links are already filtered by the link stage. Avoid
            # following every symlink to Lustre before stage 0 can start writing.
            sessions.append(path)
        elif path.is_dir() and (path / "machine_generated_transcript.json").is_file():
            sessions.append(path)
    return sorted(sessions)


def load_speaker_count_tsv(path: Path) -> dict[str, int]:
    """Load ``<count> <session_id>`` lines from a speaker-count TSV."""
    counts: dict[str, int] = {}
    if not path.is_file():
        return counts
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            counts[parts[1]] = int(parts[0])
        except ValueError:
            continue
    return counts


def load_session_id_list(path: Path) -> list[str]:
    """Load one session id per line (comments and blanks ignored)."""
    if not path.is_file():
        return []
    ids: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line.split()[0])
    return ids


def filter_sessions_by_ids(sessions: list[Path], session_ids: list[str]) -> list[Path]:
    wanted = set(session_ids)
    if not wanted:
        return sessions
    by_name = {session.name: session for session in sessions}
    missing = sorted(wanted - set(by_name))
    if missing:
        logger.warning("sessions-file: %d id(s) not found under data root (first: %s)", len(missing), missing[0])
    return [by_name[sid] for sid in session_ids if sid in by_name]


def order_sessions_by_speaker_priority(
    sessions: list[Path],
    speaker_counts: dict[str, int],
    *,
    min_priority_speakers: int,
) -> list[Path]:
    """Put sessions with at least *min_priority_speakers* first (higher counts earlier)."""
    if min_priority_speakers <= 1 or not speaker_counts:
        return sessions

    def sort_key(session_dir: Path) -> tuple[int, int, str]:
        count = speaker_counts.get(session_dir.name, 0)
        priority_bucket = 0 if count >= min_priority_speakers else 1
        return (priority_bucket, -count, session_dir.name)

    return sorted(sessions, key=sort_key)


def recording_id(speaker_id: str, session_id: str) -> str:
    return f"{speaker_id}_{session_id}_postprocessed"


def masked_speaker_audio_path(audio_masked_dir: Path, speaker_id: str, session_id: str) -> Path:
    return audio_masked_dir / f"{recording_id(speaker_id, session_id)}.wav"


def masked_speaker_rttm_path(audio_masked_dir: Path, speaker_id: str, session_id: str) -> Path:
    return audio_masked_dir / f"{recording_id(speaker_id, session_id)}.rttm"


def recording_textgrid_path(
    textgrid_dir: Path,
    recording_id: str,
    *,
    variant: str = "ordinary",
) -> Path:
    suffix = {"ordinary": "", "fastmss": "_fastmss", "fb": "_fb"}.get(variant, "")
    return textgrid_dir / f"{recording_id}{suffix}.TextGrid"


def recording_textgrid_paths(textgrid_dir: Path, recording_id: str) -> list[Path]:
    ordinary = recording_textgrid_path(textgrid_dir, recording_id, variant="ordinary")
    fb_path = recording_textgrid_path(textgrid_dir, recording_id, variant="fb")
    if ordinary.is_file() and fb_path.is_file():
        return [ordinary, fb_path]
    if ordinary.is_file():
        return [ordinary]
    if fb_path.is_file():
        return [fb_path]
    return [ordinary]


def fastmss_textgrid_path(textgrid_dir: Path, recording_id: str) -> Path:
    return recording_textgrid_path(textgrid_dir, recording_id, variant="fastmss")


def session_textgrid_path(
    textgrid_dir: Path,
    session_id: str,
    *,
    variant: str = "ordinary",
) -> Path:
    suffix = {"ordinary": "", "fastmss": "_fastmss"}.get(variant, "")
    return textgrid_dir / f"{session_id}{suffix}.TextGrid"


def interval_overlaps(start: float, end: float, intervals: list[tuple[float, float]]) -> bool:
    return any(start < interval_end and end > interval_start for interval_start, interval_end in intervals)


def speech_intervals_from_textgrid(tg_path: Path) -> list[tuple[float, float]]:
    return [(start, end) for start, end, _ in parse_textgrid_words(tg_path)]


def fb_intervals_for_recording(textgrid_dir: Path, recording_id: str) -> list[tuple[float, float]]:
    fb_path = recording_textgrid_path(textgrid_dir, recording_id, variant="fb")
    if fb_path.is_file():
        return speech_intervals_from_textgrid(fb_path)
    ordinary = recording_textgrid_path(textgrid_dir, recording_id, variant="ordinary")
    fastmss = recording_textgrid_path(textgrid_dir, recording_id, variant="fastmss")
    if ordinary.is_file() and fastmss.is_file():
        ordinary_words = parse_textgrid_words(ordinary)
        fastmss_words = parse_textgrid_words(fastmss)
        if len(ordinary_words) > len(fastmss_words):
            fastmss_intervals = [(s, e) for s, e, _ in fastmss_words]
            return [
                (start, end)
                for start, end, word in ordinary_words
                if word == "speech" and not interval_overlaps(start, end, fastmss_intervals)
            ]
    return []


def alignment_items_from_textgrid(tg_path: Path) -> list:
    from lhotse.supervision import AlignmentItem

    try:
        words = parse_textgrid_words(tg_path)
    except Exception as exc:
        log_exception(f"alignment extraction failed for {tg_path}", exc)
        return []

    items = []
    for start, end, word in words:
        items.append(
            AlignmentItem(
                symbol=word,
                start=round(start, 6),
                duration=round(end - start, 6),
            )
        )
    return items


def alignment_items_for_lhotse(
    main_tg_path: Path,
    *,
    fb_intervals: list[tuple[float, float]] | None = None,
) -> list:
    if fb_intervals is None:
        fb_intervals = fb_intervals_for_recording(main_tg_path.parent, main_tg_path.stem)

    from lhotse.supervision import AlignmentItem

    try:
        words = parse_textgrid_words(main_tg_path)
    except Exception as exc:
        log_exception(f"alignment extraction failed for {main_tg_path}", exc)
        return []

    items = []
    for start, end, word in words:
        if fb_intervals and interval_overlaps(start, end, fb_intervals):
            continue
        items.append(
            AlignmentItem(
                symbol=word,
                start=round(start, 6),
                duration=round(end - start, 6),
            )
        )
    return items


def write_lhotse_concatenated_textgrid(
    main_tg_path: Path,
    output_path: Path,
    *,
    fb_intervals: list[tuple[float, float]] | None = None,
    xmax: float | None = None,
) -> None:
    if fb_intervals is None:
        fb_intervals = fb_intervals_for_recording(main_tg_path.parent, main_tg_path.stem)

    words = parse_textgrid_words(main_tg_path)
    if fb_intervals:
        words = [
            (start, end, word)
            for start, end, word in words
            if not interval_overlaps(start, end, fb_intervals)
        ]
    write_textgrid(words, output_path, xmin=0.0, xmax=xmax)


def write_rttm(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def speech_rttm_line(
    file_id: str,
    speaker: str,
    start: float,
    end: float,
    *,
    label: str = "speech",
) -> str:
    """RTTM speech interval; *label* is written to the subtype field."""
    dur = max(end - start, 0.0)
    return (
        f"SPEAKER {file_id} 1 {start:.6f} {dur:.6f} <NA> {label} {speaker} <NA> <NA>"
    )


def build_speech_rttm_lines(
    file_id: str,
    speaker: str,
    intervals: list[tuple[float, float]],
    *,
    label: str = "speech",
    merge_gap: float = 0.0,
) -> list[str]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and (start - merged[-1][1]) <= merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return [
        speech_rttm_line(file_id, speaker, start, end, label=label)
        for start, end in merged
    ]


def normalization_log_entry(
    row: dict,
    *,
    text_raw: str | None = None,
    text_norm: str,
    num2words_lang: str,
    error: str = "",
) -> dict:
    text = text_raw if text_raw is not None else (row.get("text_raw") or row.get("text") or "")
    return {
        "session_id": row.get("session_id"),
        "speaker_id": row.get("speaker_id"),
        "recording_id": row.get("recording_id"),
        "segment_index": row.get("segment_index"),
        "start": float(row["start"]),
        "end": float(row["end"]),
        "duration": float(row.get("duration", float(row["end"]) - float(row["start"]))),
        "text": text,
        "text_norm": text_norm,
        "changed": text != text_norm,
        "num2words_lang": num2words_lang,
        "error": error,
    }


def segment_fallback_log_entry(
    seg: dict,
    recording_id: str,
    *,
    reason: str,
    detail: str = "",
) -> dict:
    return {
        "recording_id": recording_id,
        "session_id": seg.get("session_id"),
        "speaker_id": seg.get("speaker_id"),
        "segment_index": seg.get("segment_index"),
        "start": float(seg["start"]),
        "end": float(seg["end"]),
        "duration": float(seg.get("duration", float(seg["end"]) - float(seg["start"]))),
        "text_norm": seg.get("text_norm", ""),
        "reason": reason,
        "detail": detail,
        "fallback": "manifest_boundaries",
        "rttm_label": "speech",
    }


def merge_session_rttm(
    rttm_paths: list[Path],
    session_id: str,
    output_path: Path,
) -> int:
    parsed: list[tuple[float, str]] = []
    for path in rttm_paths:
        if not path.is_file():
            logger.warning("%s: missing per-recording RTTM %s", session_id, path)
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            log_exception(f"cannot read RTTM {path}", exc)
            continue
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(";"):
                continue
            parts = line.split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            try:
                start = float(parts[3])
            except ValueError:
                logger.warning("%s: invalid RTTM start time in %s: %s", session_id, path.name, line)
                continue
            parts[1] = session_id
            parsed.append((start, " ".join(parts)))

    parsed.sort(key=lambda x: x[0])
    lines = [line for _, line in parsed]
    write_rttm(output_path, lines)
    return len(lines)


def load_norm_manifest_rows(
    manifests_dir: Path,
    *,
    sessions: list[str] | None = None,
) -> tuple[list[dict], int]:
    rows: list[dict] = []
    manifest_errors = 0
    wanted = set(sessions) if sessions else None
    for path in sorted(manifests_dir.glob("*_norm.jsonl")):
        if path.name == "all_norm.jsonl":
            continue
        try:
            file_rows = load_jsonl(path)
        except Exception as exc:
            manifest_errors += 1
            log_exception(f"cannot load manifest {path}", exc)
            continue
        if wanted is not None:
            file_rows = [r for r in file_rows if r.get("session_id") in wanted]
        rows.extend(file_rows)
    return rows, manifest_errors


def group_segments_by_recording(rows: list[dict]) -> dict[str, list[dict]]:
    from collections import defaultdict

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["recording_id"]].append(row)
    for segments in grouped.values():
        segments.sort(key=lambda r: (r["start"], r["segment_index"]))
    return grouped


def group_segments_by_session(rows: list[dict]) -> dict[str, list[dict]]:
    from collections import defaultdict

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["session_id"]].append(row)
    for segments in grouped.values():
        segments.sort(key=lambda r: (r["start"], r["segment_index"]))
    return grouped


def group_recordings_by_session(rows: list[dict]) -> dict[str, list[dict]]:
    from collections import defaultdict

    grouped: dict[str, list[dict]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        session_id = row["session_id"]
        rec_id = row["recording_id"]
        if rec_id in seen[session_id]:
            continue
        seen[session_id].add(rec_id)
        grouped[session_id].append(
            {
                "recording_id": rec_id,
                "speaker_id": row["speaker_id"],
                "audio_path": Path(row["audio_filepath_16k"]),
            }
        )
    for recordings in grouped.values():
        recordings.sort(key=lambda e: e["recording_id"])
    return grouped


def load_fallback_intervals(fallback_log: Path, recording_id: str) -> list[tuple[float, float]]:
    if not fallback_log.is_file():
        return []
    intervals: list[tuple[float, float]] = []
    try:
        for row in load_jsonl(fallback_log):
            if row.get("recording_id") != recording_id:
                continue
            intervals.append((float(row["start"]), float(row["end"])))
    except Exception as exc:
        log_exception(f"cannot read fallback log for {recording_id}", exc)
    return intervals


def alignment_items_from_words(words: list[tuple[float, float, str]]) -> list:
    from lhotse.supervision import AlignmentItem

    return [
        AlignmentItem(
            symbol=word,
            start=round(start, 6),
            duration=round(end - start, 6),
        )
        for start, end, word in words
    ]


def words_to_json(words: list[tuple[float, float, str]]) -> list[list]:
    return [[start, end, word] for start, end, word in words]


def words_from_json(raw: list) -> list[tuple[float, float, str]]:
    return [(float(s), float(e), str(w)) for s, e, w in raw]


def tagged_words_to_json(words: list[tuple[float, float, str, str]]) -> list[list]:
    return [[start, end, word, speaker_id] for start, end, word, speaker_id in words]


def tagged_words_from_json(raw: list) -> list[tuple[float, float, str, str]]:
    out: list[tuple[float, float, str, str]] = []
    for item in raw:
        if len(item) == 4:
            s, e, w, spk = item
            out.append((float(s), float(e), str(w), str(spk)))
        else:
            s, e, w = item
            out.append((float(s), float(e), str(w), ""))
    return out


def alignment_record(
    recording_id: str,
    segments: list[dict],
    *,
    merged_words: list[tuple[float, float, str]],
    fb_words: list[tuple[float, float, str]],
    audio_duration: float,
) -> dict:
    return {
        "recording_id": recording_id,
        "speaker_id": segments[0]["speaker_id"],
        "session_id": segments[0]["session_id"],
        "audio_filepath_16k": segments[0]["audio_filepath_16k"],
        "audio_duration": audio_duration,
        "merged_words": words_to_json(merged_words),
        "fb_words": words_to_json(fb_words),
    }


def session_alignment_record(
    session_id: str,
    *,
    merged_words: list[tuple[float, float, str, str]],
    fb_words: list[tuple[float, float, str, str]],
    audio_duration: float,
    recordings: list[dict],
) -> dict:
    return {
        "session_id": session_id,
        "audio_duration": audio_duration,
        "merged_words": tagged_words_to_json(merged_words),
        "fb_words": tagged_words_to_json(fb_words),
        "recordings": recordings,
    }


def append_alignment_record(
    path: Path,
    record: dict,
    *,
    lock: threading.Lock | None = None,
) -> None:
    append_jsonl(path, record, lock=lock)


def load_alignment_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    ids: set[str] = set()
    for row in load_jsonl(path):
        if "session_id" in row:
            ids.add(row["session_id"])
        elif "recording_id" in row:
            ids.add(row["recording_id"])
    return ids


def load_alignments_by_session(path: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    if not path.is_file():
        return by_id
    for row in load_jsonl(path):
        if "session_id" in row:
            by_id[row["session_id"]] = row
    return by_id


def load_alignments_by_recording(path: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    if not path.is_file():
        return by_id
    for row in load_jsonl(path):
        if "recordings" in row:
            for rec in row["recordings"]:
                by_id[rec["recording_id"]] = rec
        elif "recording_id" in row:
            by_id[row["recording_id"]] = row
    return by_id


def build_rttm_lines_from_words(
    recording_id: str,
    speaker_id: str,
    merged_words: list[tuple[float, float, str]],
    fb_words: list[tuple[float, float, str]],
    *,
    merge_gap: float = 0.2,
) -> list[str]:
    tagged: list[tuple[float, float, str]] = []
    for start, end, _ in merged_words:
        tagged.append((start, end, "<NA>"))
    for start, end, _ in fb_words:
        tagged.append((start, end, "speech"))
    merged = merge_tagged_speech_intervals(tagged, merge_gap)
    return [
        speech_rttm_line(recording_id, speaker_id, start, end, label=label)
        for start, end, label in merged
    ]


def build_session_rttm_lines_from_words(
    session_id: str,
    merged_words: list[tuple[float, float, str, str]],
    fb_words: list[tuple[float, float, str, str]],
    *,
    merge_gap: float = 0.2,
) -> list[str]:
    from collections import defaultdict

    by_speaker: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for start, end, _, speaker_id in merged_words:
        by_speaker[speaker_id].append((start, end, "<NA>"))
    for start, end, _, speaker_id in fb_words:
        by_speaker[speaker_id].append((start, end, "speech"))

    lines: list[tuple[float, str]] = []
    for speaker_id, tagged in by_speaker.items():
        merged = merge_tagged_speech_intervals(tagged, merge_gap)
        for start, end, label in merged:
            line = speech_rttm_line(session_id, speaker_id, start, end, label=label)
            lines.append((start, line))
    lines.sort(key=lambda x: x[0])
    return [line for _, line in lines]


def merge_session_rttm_from_line_lists(
    session_id: str,
    line_lists: list[list[str]],
) -> list[str]:
    parsed: list[tuple[float, str]] = []
    for lines in line_lists:
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith(";"):
                continue
            parts = line.split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            try:
                start = float(parts[3])
            except ValueError:
                continue
            parts[1] = session_id
            parsed.append((start, " ".join(parts)))
    parsed.sort(key=lambda x: x[0])
    return [line for _, line in parsed]


def build_recording_rttm_lines(
    recording_id: str,
    speaker_id: str,
    tg_path: Path,
    *,
    fallback_log: Path | None = None,
    merge_gap: float = 0.2,
) -> list[str]:
    textgrid_dir = tg_path.parent
    tagged: list[tuple[float, float, str]] = []
    for path in recording_textgrid_paths(textgrid_dir, recording_id):
        try:
            for start, end, word in parse_textgrid_words(path):
                label = "speech" if word == "speech" else "<NA>"
                tagged.append((start, end, label))
        except Exception as exc:
            log_exception(f"RTTM conversion failed for {path}", exc)

    fb_path = recording_textgrid_path(textgrid_dir, recording_id, variant="fb")
    if fallback_log is not None and not fb_path.is_file():
        for start, end in load_fallback_intervals(fallback_log, recording_id):
            tagged.append((start, end, "speech"))

    merged = merge_tagged_speech_intervals(tagged, merge_gap)
    return [
        speech_rttm_line(recording_id, speaker_id, start, end, label=label)
        for start, end, label in merged
    ]


def pad_speech_intervals(
    speech_intervals: list[tuple[float, float]],
    pad: float,
    duration: float,
) -> list[tuple[float, float]]:
    """Grow each speech interval by *pad* seconds on both sides, clamped to [0, duration].

    Overlaps created by padding are merged. Used to keep a margin of untouched
    audio around speech boundaries so pause noise never abuts speech.
    """
    if pad <= 0:
        return merge_speech_intervals(speech_intervals, 0.0)
    padded = [
        (max(0.0, start - pad), min(duration, end + pad))
        for start, end in speech_intervals
    ]
    return merge_speech_intervals(padded, 0.0)


def invert_intervals(
    speech_intervals: list[tuple[float, float]],
    duration: float,
) -> list[tuple[float, float]]:
    """Return gaps between *speech_intervals* over [0, *duration*] (pause regions)."""
    pauses: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in sorted(speech_intervals):
        if start > cursor + 1e-6:
            pauses.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration - 1e-6:
        pauses.append((cursor, duration))
    return pauses


def parse_rttm_speech_intervals(
    lines: list[str],
    *,
    merge_gap: float = 0.2,
) -> list[tuple[float, float]]:
    """Speech intervals from RTTM lines (<NA> and speech subtype labels)."""
    raw: list[tuple[float, float]] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        label = parts[6]
        if label not in {"<NA>", "speech"}:
            continue
        try:
            start = float(parts[3])
            dur = float(parts[4])
        except ValueError:
            continue
        if dur > 0:
            raw.append((start, start + dur))
    return merge_speech_intervals(raw, merge_gap)


def parse_session_rttm_by_speaker(
    lines: list[str],
    *,
    merge_gap: float = 0.2,
) -> dict[str, list[tuple[float, float]]]:
    """Speech intervals per speaker from a session-level RTTM (stage 4 output)."""
    from collections import defaultdict

    by_speaker: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        label = parts[6]
        if label not in {"<NA>", "speech"}:
            continue
        speaker_id = parts[7]
        try:
            start = float(parts[3])
            dur = float(parts[4])
        except ValueError:
            continue
        if dur > 0:
            by_speaker[speaker_id].append((start, start + dur))

    return {
        speaker_id: merge_speech_intervals(intervals, merge_gap)
        for speaker_id, intervals in by_speaker.items()
    }


def session_rttm_path(audio_mixed_dir: Path, session_id: str) -> Path:
    return audio_mixed_dir / f"{session_id}.rttm"


def load_session_rttm_by_speaker(
    rttm_path: Path,
    *,
    merge_gap: float = 0.2,
) -> dict[str, list[tuple[float, float]]]:
    if not rttm_path.is_file():
        return {}
    lines = rttm_path.read_text(encoding="utf-8").splitlines()
    return parse_session_rttm_by_speaker(lines, merge_gap=merge_gap)


def speech_intervals_from_recording_alignment(
    rec_row: dict,
    *,
    merge_gap: float = 0.2,
) -> list[tuple[float, float]]:
    merged = words_from_json(rec_row.get("merged_words", []))
    fb = words_from_json(rec_row.get("fb_words", []))
    raw = [(start, end) for start, end, _ in merged] + [(start, end) for start, end, _ in fb]
    return merge_speech_intervals(raw, merge_gap)


def decode_audio_mono_f32(path: Path, *, target_sr: int = 16000) -> tuple:
    import numpy as np

    cmd = [
        ffmpeg_executable(),
        "-nostdin",
        "-i",
        str(path),
        "-ar",
        str(target_sr),
        "-ac",
        "1",
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=FFMPEG_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffmpeg decode failed to start for %s: %s", path, exc)
        raise
    if result.returncode != 0:
        msg = result.stderr.decode(errors="replace")[-400:]
        msg_0 = f"ffmpeg decode failed for {path}: {msg}"
        raise RuntimeError(msg_0)
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return audio, target_sr


def encode_audio_mono_f32_to_wav(
    audio,
    dst: Path,
    *,
    sample_rate: int = 16000,
) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_executable(),
        "-y",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    try:
        result = subprocess.run(
            cmd,
            input=audio.tobytes(),
            capture_output=True,
            check=False,
            timeout=FFMPEG_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffmpeg WAV encode failed to start for %s: %s", dst, exc)
        return False
    if result.returncode != 0:
        logger.warning(
            "ffmpeg WAV encode failed for %s: %s",
            dst,
            result.stderr.decode(errors="replace")[-400:],
        )
        return False
    return True


def apply_white_noise_in_pause_intervals(
    src: Path,
    dst: Path,
    pause_intervals: list[tuple[float, float]],
    *,
    target_sr: int = 16000,
    noise_level: float = 0.0002,
    seed: int | None = None,
    preserve_speech: bool = True,
    stitch_ms: float = 5.0,
) -> bool:
    """Replace *pause_intervals* with white noise; RTTM speech samples stay intact.

    Speech regions are never modified. Only samples inside each pause interval are
    written.     When *preserve_speech* is True and *stitch_ms* > 0, a linear
    crossfade is applied **inside** the pause at both ends: original pause audio
    is blended into white noise at the pause start and blended back out before the
    pause end (still within the pause interval).

    When *preserve_speech* is False, the pause interior is filled with noise
    with no crossfade.
    """
    import numpy as np

    try:
        audio, sr = decode_audio_mono_f32(src, target_sr=target_sr)
    except RuntimeError as exc:
        log_exception(f"decode failed for {src}", exc)
        return False

    audio = np.array(audio, dtype=np.float32, copy=True)
    n = len(audio)
    rng = np.random.default_rng(seed)
    stitch = max(0, round(stitch_ms * sr / 1000.0)) if preserve_speech else 0

    for start, end in pause_intervals:
        i0 = max(0, int(start * sr))
        i1 = min(n, int(end * sr))
        length = i1 - i0
        if length <= 0:
            continue

        orig_pause = audio[i0:i1].copy()
        noise = rng.standard_normal(length, dtype=np.float32) * noise_level

        if stitch > 0:
            fade = min(stitch, length // 2)
            if fade > 0:
                ramp_in = np.linspace(0.0, 1.0, fade, dtype=np.float32)
                ramp_out = np.linspace(1.0, 0.0, fade, dtype=np.float32)
                noise[:fade] = orig_pause[:fade] * (1.0 - ramp_in) + noise[:fade] * ramp_in
                noise[-fade:] = orig_pause[-fade:] * ramp_out + noise[-fade:] * (1.0 - ramp_out)

        audio[i0:i1] = noise

    return encode_audio_mono_f32_to_wav(audio, dst, sample_rate=sr)


def prepare_speaker_audio_for_session_mix(
    audio_path: Path,
    dst: Path,
    *,
    speech_intervals: list[tuple[float, float]],
    audio_duration: float | None = None,
    noise_level: float = 0.0002,
    seed: int | None = None,
    preserve_speech: bool = True,
    stitch_ms: float = 5.0,
    boundary_indent: float = 0.5,
) -> bool:
    """Fill non-speech (pause) regions with white noise before session mixing.

    *boundary_indent* keeps that many seconds of original audio on each side of a
    speech interval untouched (pause noise starts 0.5s after speech ends and stops
    0.5s before speech begins by default).
    """
    if audio_duration is None:
        try:
            audio_duration = ffprobe_duration(audio_path)
        except RuntimeError:
            audio_duration = max((end for _, end in speech_intervals), default=0.0) + 0.01

    padded_speech = pad_speech_intervals(speech_intervals, boundary_indent, audio_duration)
    pause_intervals = invert_intervals(padded_speech, audio_duration)
    return apply_white_noise_in_pause_intervals(
        audio_path,
        dst,
        pause_intervals,
        noise_level=noise_level,
        seed=seed,
        preserve_speech=preserve_speech,
        stitch_ms=stitch_ms,
    )


def session_mixed_audio_path(audio_mixed_dir: Path, session_id: str) -> Path:
    return audio_mixed_dir / f"{session_id}.wav"


def mix_audio_files(audio_paths: list[Path], output_path: Path) -> bool:
    existing = [p for p in audio_paths if p.is_file()]
    if not existing:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(existing) == 1:
        import shutil

        try:
            shutil.copy2(existing[0], output_path)
            return True
        except OSError as exc:
            log_exception(f"cannot copy mixed audio to {output_path}", exc)
            return False

    cmd = [ffmpeg_executable(), "-nostdin", "-y"]
    for path in existing:
        cmd.extend(["-i", str(path)])
    n = len(existing)
    cmd.extend(
        [
            "-filter_complex",
            f"amix=inputs={n}:duration=longest:dropout_transition=0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(output_path),
        ]
    )
    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=FFMPEG_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffmpeg mix failed to start for %s: %s", output_path.name, exc)
        return False
    if result.returncode != 0:
        logger.warning("ffmpeg mix failed for %s: %s", output_path.name, result.stderr[-400:])
        return False
    return True


def run_main(main_fn) -> None:
    """Entry-point wrapper: log tracebacks and return non-zero exit codes."""
    try:
        raise SystemExit(main_fn())
    except PipelineError as exc:
        logger.exception("Pipeline failed")
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        logger.exception("Interrupted")
        raise SystemExit(130) from None
    except Exception as exc:
        logger.exception("Unhandled error")
        raise SystemExit(1) from exc
