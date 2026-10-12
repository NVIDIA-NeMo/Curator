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

import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from nemo_curator.stages.audio.tts.early_cutoff.mfa_resolver import (
    MODEL_UNAVAILABLE_REVIEW,
    TECHNICAL_ERROR_REVIEW,
    UNSUPPORTED_LANGUAGE_REVIEW,
    EarlyCutOffMFAResolverStage,
    calibration_thresholds,
    is_candidate,
    is_resolved,
    resolve_candidate,
)
from nemo_curator.stages.audio.tts.early_cutoff.mfa_runtime import (
    build_align_command,
    discover_database,
    discover_textgrids,
    extract_textgrid_evidence,
    parse_long_textgrid,
    safe_name,
    sqlite_likelihood_evidence,
)
from nemo_curator.stages.audio.tts.early_cutoff.verifier import EVIDENCE_NAMES, raw_evidence, verifier_components
from nemo_curator.tasks import AudioTask

NULL = {name: [0.0, 0.1, 0.2, 0.3] for name in EVIDENCE_NAMES}
SR = 16000


# ----------------------------------------------------------------------------- fixtures / fakes


def gate(language: str = "en", *, strong: bool = True, candidate: bool = True) -> dict[str, Any]:
    """Gate annotation as written by EarlyCutOffEndpointGateStage (only the fields the resolver reads)."""
    return {
        "schema_version": 13,
        "cascade_version": "endpoint_mfa_v13",
        "cascade_step": "gpu_endpoint_gate",
        "analysis_status": "OK",
        "annotation_only": True,
        "language": language,
        "decision": "REVIEW" if candidate else "OK",
        "mfa_verification_required": candidate,
        "is_suspicious": candidate,
        "verifier_status": "PENDING" if candidate else "NOT_INVOKED",
        "endpoint_probability": 0.99 if candidate else 0.1,
        "terminal_partial_ctc_log_likelihood_ratio": 5.0 if strong else -50.0,
        "terminal_censor_log_odds": 5.0 if strong else -50.0,
        "components": {"suffix_non_recovery": 1.0 if strong else 0.0},
        "last_phone_confidence": 0.0 if strong else 1.0,
        "eou_cutoff_score": 1.0 if strong else 0.0,
        "whole_phone_cut_score": 0.0,
        "endpoint_bad_region_score": 0.0,
    }


def write_wav(path: Path, seconds: float = 1.0) -> Path:
    t = np.arange(int(seconds * SR)) / SR
    sf.write(path, (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), SR)
    return path


def segment(tmp_path: Path, name: str, annotation: dict[str, Any] | None, text: str = "hello world") -> dict:
    seg: dict[str, Any] = {"audio_filepath": str(write_wav(tmp_path / f"{name}.wav")), "GranaryV2": {"tn_raw": text}}
    if annotation is not None:
        seg["early_cut_off_detection"] = annotation
    return seg


def textgrid(duration: float, phone_end: float) -> str:
    def tier(name: str, intervals: list[tuple[float, float, str]]) -> str:
        body = "".join(
            f'        intervals [{i}]:\n            xmin = {a}\n            xmax = {b}\n            text = "{t}"\n'
            for i, (a, b, t) in enumerate(intervals, 1)
        )
        return (
            f'        class = "IntervalTier"\n        name = "{name}"\n        xmin = 0\n        xmax = {duration}\n'
            f"        intervals: size = {len(intervals)}\n{body}"
        )

    words = [(0.0, 0.1, ""), (0.1, phone_end, "world"), (phone_end, duration, "")]
    phones = [(0.0, 0.1, "sil"), (0.1, phone_end - 0.1, "W"), (phone_end - 0.1, phone_end, "D")]
    if phone_end < duration:
        phones.append((phone_end, duration, "sp"))
    return (
        'File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = 0\nxmax = '
        f"{duration}\ntiers? <exists>\nsize = 2\nitem []:\n    item [1]:\n{tier('words', words)}"
        f"    item [2]:\n{tier('phones', phones)}"
    )


def create_mfa_db(path: Path, utterances: dict[str, float], *, final_goodness: float = -20.0) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE file (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE utterance (id INTEGER PRIMARY KEY, file_id INT, alignment_log_likelihood REAL,
            speech_log_likelihood REAL, snr REAL, duration_deviation REAL, max_running_short_interval REAL);
        CREATE TABLE phone (id INTEGER PRIMARY KEY, phone TEXT);
        CREATE TABLE phone_interval (id INTEGER PRIMARY KEY, utterance_id INT, phone_id INT, begin REAL, end REAL,
            phone_goodness REAL, intensity REAL);
        INSERT INTO phone VALUES (1, 'W'), (2, 'ER'), (3, 'L'), (4, 'D');
        """
    )
    for uid, (name, end) in enumerate(utterances.items(), 1):
        con.execute("INSERT INTO file VALUES (?, ?)", (uid, name))
        con.execute("INSERT INTO utterance VALUES (?, ?, -5.0, -6.0, 20.0, 0.1, 2.0)", (uid, uid))
        for pid, (begin, stop, goodness) in enumerate(
            [(0.1, 0.3, -2.0), (0.3, 0.5, -3.0), (0.5, end - 0.1, -4.0), (end - 0.1, end, final_goodness)], 1
        ):
            con.execute(
                "INSERT INTO phone_interval (utterance_id, phone_id, begin, end, phone_goodness, intensity) "
                "VALUES (?, ?, ?, ?, ?, 60.0)",
                (uid, pid, begin, stop, goodness),
            )
    con.commit()
    con.close()


class FakeMFA:
    """Stands in for ``subprocess.run(["mfa", "align", corpus, dict, model, out, "--temporary_directory", tmp, ...])``.

    Writes one TextGrid per corpus utterance and an MFA-like SQLite database, like MFA 3.x with --no_final_clean.
    """

    def __init__(
        self,
        *,
        touches_eof: bool = True,
        skip: set[str] | None = None,
        fail: bool = False,
        mfa_root: Path | None = None,
    ) -> None:
        self.calls: list[list[str]] = []
        self.envs: list[dict | None] = []
        self.touches_eof = touches_eof
        self.skip = skip or set()
        self.fail = fail
        self.mfa_root = mfa_root

    def _extract_models(self, dictionary: str, acoustic: str) -> None:
        """Like MFA's first use of a pretrained model name: extract it under MFA_ROOT_DIR."""
        if self.mfa_root is None or Path(dictionary).exists():
            return
        (self.mfa_root / "pretrained_models/dictionary").mkdir(parents=True, exist_ok=True)
        (self.mfa_root / f"pretrained_models/dictionary/{dictionary}.dict").write_text("WORLD W ER L D\n")
        model_dir = self.mfa_root / f"extracted_models/acoustic/{acoustic}_acoustic"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "final.mdl").write_text("model\n")
        (model_dir / "tree").write_text("tree\n")

    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        self.calls.append(command)
        self.envs.append(kwargs.get("env"))
        if self.fail:
            raise subprocess.CalledProcessError(1, command, stderr="kaldi exploded")
        corpus, out, tmp = (
            Path(command[2]),
            Path(command[5]),
            Path(command[command.index("--temporary_directory") + 1]),
        )
        self._extract_models(command[3], command[4])
        ends: dict[str, float] = {}
        for wav in sorted((corpus / "candidate").glob("*.wav")):
            if wav.stem in self.skip:
                continue
            duration = sf.info(str(wav)).duration
            end = duration if self.touches_eof else duration - 0.4
            (out / "candidate").mkdir(parents=True, exist_ok=True)
            (out / "candidate" / f"{wav.stem}.TextGrid").write_text(textgrid(duration, end), encoding="utf-8")
            ends[wav.stem] = end
        (tmp / "corpus").mkdir(parents=True, exist_ok=True)
        create_mfa_db(tmp / "corpus" / "corpus.db", ends)
        return subprocess.CompletedProcess(command, 0, "", "")


@pytest.fixture
def calibration_dir(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    root.mkdir()
    for name in ("english_verifier.json", "multilingual_verifier.json"):
        (root / name).write_text(
            json.dumps({"null_evidence": NULL, "thresholds": {"verifier_threshold": 0.5}}), encoding="utf-8"
        )
    (root / "multilingual_thresholds.json").write_text(
        json.dumps(
            {
                "global_thresholds": {"verifier_threshold": 0.5},
                "language_overrides": {"it": {"verifier_threshold": 2.0}},
            }
        ),
        encoding="utf-8",
    )
    return root


def stage(calibration_dir: Path, runner: FakeMFA | None, **kwargs: object) -> EarlyCutOffMFAResolverStage:
    return EarlyCutOffMFAResolverStage(bundle_dir=str(calibration_dir), command_runner=runner, **kwargs)


def annotation(task: AudioTask, index: int) -> dict[str, Any]:
    return task.data["segments"][index]["early_cut_off_detection"]


# ----------------------------------------------------------------------------- pure helpers


def test_verifier_scores_strong_candidate_high_and_weak_low() -> None:
    mfa = {
        "touches_eof": True,
        "likelihood_status": "AVAILABLE",
        "final_vs_local_loglike": -10.0,
        "final_vs_alignment_loglike": -10.0,
        "final_phone_duration_sec": 0.0,
        "max_running_short_interval": 5.0,
    }
    strong = verifier_components({"prediction": gate(strong=True), "mfa": mfa}, NULL)
    assert strong["valid_mfa_endpoint"] is True
    assert strong["ctc_score"] == 1.0
    assert strong["score"] == pytest.approx(1.0)
    weak = verifier_components({"prediction": gate(strong=False), "mfa": mfa}, NULL)
    assert weak["ctc_score"] == 0.0
    assert weak["score"] == 0.0
    # without an EOF-touching MFA endpoint only the whole-phone path can fire
    detached = verifier_components({"prediction": gate(strong=True), "mfa": {**mfa, "touches_eof": False}}, NULL)
    assert detached["partial_score"] == 0.0


def test_raw_evidence_defaults_for_missing_values() -> None:
    raw = raw_evidence({})
    assert raw["ctc_age"] == -100.0
    assert raw["low_last_phone_confidence"] == 0.0
    assert raw["mfa_short_duration"] == 0.0


def test_verifier_rejects_empty_null_channel() -> None:
    with pytest.raises(ValueError, match="empty"):
        verifier_components({}, {**NULL, "eou": []})


def test_resolve_candidate_threshold_and_missing_likelihood() -> None:
    mfa = {"touches_eof": True, "likelihood_status": "AVAILABLE", "max_running_short_interval": 5.0}
    assert resolve_candidate(gate(), dict(mfa), NULL, 0.5)["decision"] == "DROP"
    resolved = resolve_candidate(gate(), dict(mfa), NULL, 1.0)
    assert resolved["decision"] == "OK"
    assert resolved["verifier_status"] == "AVAILABLE"
    assert resolved["verifier_threshold"] == 1.0
    review = resolve_candidate(gate(), {"likelihood_status": "UNAVAILABLE"}, NULL, 0.5)
    assert review["decision"] == "REVIEW"
    assert review["verifier_status"] == TECHNICAL_ERROR_REVIEW
    assert review["mfa"] == {"likelihood_status": "UNAVAILABLE"}


def test_calibration_thresholds_language_override() -> None:
    calibration = {
        "global_thresholds": {"verifier_threshold": 0.5},
        "language_overrides": {"it": {"verifier_threshold": 2}},
    }
    assert calibration_thresholds(calibration, "de")["verifier_threshold"] == 0.5
    assert calibration_thresholds(calibration, "it")["verifier_threshold"] == 2
    assert calibration_thresholds({"thresholds": {"verifier_threshold": 0.7}}, "en")["verifier_threshold"] == 0.7


def test_candidate_and_resolved_predicates() -> None:
    assert is_candidate(gate())
    assert not is_candidate(gate(candidate=False))
    assert not is_candidate({**gate(), "schema_version": 12})
    assert is_resolved({**gate(), "cascade_step": "mfa_resolver", "decision": "DROP"})
    assert not is_resolved({**gate(), "cascade_step": "mfa_resolver"})  # REVIEW is never final
    assert not is_resolved(None)


def test_textgrid_parsing_and_evidence(tmp_path: Path) -> None:
    path = tmp_path / "u.TextGrid"
    path.write_text(textgrid(2.0, 1.5), encoding="utf-8")
    tiers = parse_long_textgrid(path)
    assert [t["name"] for t in tiers] == ["words", "phones"]
    evidence = extract_textgrid_evidence(path, audio_duration_sec=2.0, eof_threshold_sec=0.03)
    assert evidence["final_word_label"] == "world"
    assert evidence["final_phone_label"] == "D"  # trailing "sp" is silence
    assert evidence["final_phone_duration_sec"] == pytest.approx(0.1)
    assert evidence["end_to_audio_eof_sec"] == pytest.approx(0.5)
    assert evidence["touches_eof"] is False
    assert extract_textgrid_evidence(path, audio_duration_sec=1.51, eof_threshold_sec=0.03)["touches_eof"] is True
    empty = tmp_path / "empty.TextGrid"
    empty.write_text('File type = "ooTextFile"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no long TextGrid"):
        parse_long_textgrid(empty)


def test_sqlite_likelihood_evidence(tmp_path: Path) -> None:
    db = tmp_path / "corpus.db"
    create_mfa_db(db, {"Utt_A": 1.0, "utt_b": 2.0})
    evidence = sqlite_likelihood_evidence(db, ["utt_a", "missing"])
    assert evidence["missing"] == {"likelihood_status": "UNAVAILABLE"}
    a = evidence["utt_a"]
    assert a["likelihood_status"] == "AVAILABLE"
    assert a["final_phone_db_label"] == "D"
    assert a["final_phone_path_loglike"] == -20.0
    assert a["final_vs_alignment_loglike"] == -15.0
    assert a["final_vs_local_loglike"] == -17.0  # -20 - median(-2, -3, -4)
    assert a["max_running_short_interval"] == 2.0
    assert discover_database(tmp_path) == db


def test_discover_textgrids_drops_ambiguous(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    for p in (tmp_path / "a" / "X.TextGrid", tmp_path / "b" / "x.TextGrid", tmp_path / "a" / "y.TextGrid"):
        p.write_text("", encoding="utf-8")
    assert set(discover_textgrids(tmp_path)) == {"y"}


def test_build_align_command_default_and_template(tmp_path: Path) -> None:
    command = build_align_command("mfa", tmp_path / "c", "d", "m", tmp_path / "o", tmp_path / "t", num_jobs=4)
    assert command[:6] == ["mfa", "align", str(tmp_path / "c"), "d", "m", str(tmp_path / "o")]
    assert command[command.index("--num_jobs") : command.index("--num_jobs") + 2] == ["--num_jobs", "4"]
    assert "--clean" in command
    assert "--no_final_clean" in command
    templated = build_align_command(
        "mfa", tmp_path / "c", "d", "m", tmp_path / "o", tmp_path / "t", command_template=["x", "{dictionary}"]
    )
    assert templated == ["x", "d"]
    assert safe_name("a/b c:d") == "a_b_c_d"
    assert safe_name("///") == "utterance"


# ----------------------------------------------------------------------------- stage


def test_stage_resolves_all_branches(tmp_path: Path, calibration_dir: Path) -> None:
    missing_audio = {
        "audio_filepath": str(tmp_path / "nope.wav"),
        "GranaryV2": {"tn_raw": "text"},
        "early_cut_off_detection": gate("de"),
    }
    task = AudioTask(
        data={
            "segments": [
                segment(tmp_path, "drop", gate("en", strong=True)),
                segment(tmp_path, "ok", gate("en", strong=False)),
                segment(tmp_path, "fi", gate("fi")),
                segment(tmp_path, "mt", gate("mt")),
                missing_audio,
                segment(tmp_path, "notcand", gate("en", candidate=False)),
                segment(tmp_path, "noann", None),
                segment(tmp_path, "it", gate("it", strong=True)),
                segment(tmp_path, "notext", gate("de"), text=" "),
            ]
        }
    )
    runner = FakeMFA(mfa_root=tmp_path / "mfa_root")
    resolver = stage(calibration_dir, runner, mfa_root_dir=str(tmp_path / "mfa_root"))
    [out] = resolver.process_batch([task])

    # one MFA run per language with candidates (en, it)
    assert sorted(call[3] for call in runner.calls) == ["english_us_arpa", "italian_cv"]
    assert runner.envs[0]["MFA_ROOT_DIR"] == str(tmp_path / "mfa_root")
    assert runner.envs[0]["PATH"].startswith(str(tmp_path / "mfa_root" / "runtime" / "bin"))

    drop = annotation(out, 0)
    assert drop["decision"] == "DROP"
    assert drop["verifier_status"] == "AVAILABLE"
    assert drop["is_suspicious"] is True
    assert drop["mfa"]["alignment_status"] == "ALIGNED"
    assert drop["mfa"]["touches_eof"] is True
    assert drop["mfa"]["likelihood_status"] == "AVAILABLE"
    assert "textgrid_path" not in drop["mfa"]
    assert drop["cascade_step"] == "mfa_resolver"
    assert drop["endpoint_probability"] == 0.99  # gate fields preserved

    assert annotation(out, 1)["decision"] == "OK"
    assert annotation(out, 1)["is_suspicious"] is False

    fi = annotation(out, 2)
    assert (fi["decision"], fi["verifier_status"]) == ("REVIEW", MODEL_UNAVAILABLE_REVIEW)
    assert fi["mfa"]["likelihood_status"] == "MODEL_UNAVAILABLE"
    assert fi["mfa"]["error"]["code"] == "mfa_model_unavailable"
    assert fi["verification_status"] == "MFA_MODEL_UNAVAILABLE"

    mt = annotation(out, 3)
    assert (mt["decision"], mt["verifier_status"]) == ("REVIEW", UNSUPPORTED_LANGUAGE_REVIEW)
    assert mt["verification_status"] == "UNSUPPORTED_LANGUAGE"

    for index in (4, 8):
        failed = annotation(out, index)
        assert (failed["decision"], failed["verifier_status"]) == ("REVIEW", TECHNICAL_ERROR_REVIEW)
        assert failed["mfa"]["error"]["code"] == "mfa_candidate_materialization_failed"
        assert failed["mfa"]["alignment_status"] == "ERROR"

    not_candidate = annotation(out, 5)
    assert not_candidate["decision"] == "OK"
    assert not_candidate["verifier_status"] == "NOT_INVOKED"
    assert not_candidate["cascade_step"] == "mfa_resolver"
    assert "early_cut_off_detection" not in out.data["segments"][6]

    # it has a language override threshold of 2.0 -> even a strong candidate is OK
    assert annotation(out, 7)["decision"] == "OK"
    assert annotation(out, 7)["verifier_threshold"] == 2.0


def test_stage_is_idempotent(tmp_path: Path, calibration_dir: Path) -> None:
    task = AudioTask(data={"segments": [segment(tmp_path, "a", gate("en")), segment(tmp_path, "b", gate("fi"))]})
    runner = FakeMFA()
    resolver = stage(calibration_dir, runner)
    resolver.process_batch([task])
    first = json.loads(json.dumps(task.data))
    assert len(runner.calls) == 1
    resolver.process_batch([task])
    # DROP is final -> no new alignment; the REVIEW (no model) is retried but resolves identically
    assert len(runner.calls) == 1
    assert task.data == first
    rerun = stage(calibration_dir, runner, overwrite=True)
    rerun.process_batch([task])
    assert len(runner.calls) == 2


def test_stage_flat_rows_and_batches_across_tasks(tmp_path: Path, calibration_dir: Path) -> None:
    tasks = [AudioTask(data=segment(tmp_path, f"flat{i}", gate("en"))) for i in range(3)]
    runner = FakeMFA()
    out = stage(calibration_dir, runner).process_batch(tasks)
    assert len(runner.calls) == 1  # one alignment for the whole batch
    assert len(list((Path(runner.calls[0][2]) / "candidate").glob("*.wav"))) == 0  # temp dir cleaned
    assert [t.data["early_cut_off_detection"]["decision"] for t in out] == ["DROP"] * 3


def test_stage_mfa_failure_and_missing_textgrid(tmp_path: Path, calibration_dir: Path) -> None:
    task = AudioTask(data={"segments": [segment(tmp_path, "a", gate("en")), segment(tmp_path, "b", gate("en"))]})
    stage(calibration_dir, FakeMFA(fail=True)).process_batch([task])
    for index in (0, 1):
        failed = annotation(task, index)
        assert failed["verifier_status"] == TECHNICAL_ERROR_REVIEW
        assert failed["mfa"]["error"]["code"] == "mfa_runtime_failed"
        assert "kaldi exploded" in failed["mfa"]["error"]["message"]

    task2 = AudioTask(data={"segments": [segment(tmp_path, "c", gate("en")), segment(tmp_path, "d", gate("en"))]})
    runner = FakeMFA(skip={"t0_s1_d"})
    stage(calibration_dir, runner).process_batch([task2])
    assert annotation(task2, 0)["decision"] == "DROP"
    assert annotation(task2, 1)["mfa"]["error"]["code"] == "mfa_textgrid_missing"


def test_stage_not_touching_eof_and_textgrid_dir(tmp_path: Path, calibration_dir: Path) -> None:
    task = AudioTask(data={"segments": [segment(tmp_path, "a", gate("en"))]})
    grids = tmp_path / "grids"
    stage(calibration_dir, FakeMFA(touches_eof=False), textgrid_dir=str(grids)).process_batch([task])
    resolved = annotation(task, 0)
    assert resolved["mfa"]["touches_eof"] is False
    assert resolved["verifier"]["partial_score"] == 0.0
    assert resolved["decision"] == "OK"
    assert Path(resolved["mfa"]["textgrid_path"]).is_file()
    assert Path(resolved["mfa"]["textgrid_path"]).parent == grids


def test_stage_unavailable_executable_and_calibration(tmp_path: Path, calibration_dir: Path) -> None:
    task = AudioTask(data={"segments": [segment(tmp_path, "a", gate("en"))]})
    EarlyCutOffMFAResolverStage(bundle_dir=str(calibration_dir), mfa_command="/nonexistent/mfa").process_batch([task])
    unavailable = annotation(task, 0)
    assert unavailable["verifier_status"] == MODEL_UNAVAILABLE_REVIEW
    assert unavailable["mfa"]["error"]["code"] == "mfa_executable_unavailable"

    task2 = AudioTask(data={"segments": [segment(tmp_path, "b", gate("en"))]})
    runner = FakeMFA()
    EarlyCutOffMFAResolverStage(bundle_dir=str(tmp_path / "missing"), command_runner=runner).process_batch([task2])
    assert annotation(task2, 0)["mfa"]["error"]["code"] == "verifier_calibration_unavailable"
    assert runner.calls == []


def test_stage_incomplete_model_and_custom_model_map(tmp_path: Path, calibration_dir: Path) -> None:
    task = AudioTask(data={"segments": [segment(tmp_path, "a", gate("en")), segment(tmp_path, "b", gate("de"))]})
    runner = FakeMFA()
    resolver = stage(calibration_dir, runner, mfa_models={"en": {"dictionary": "english_us_arpa"}})
    resolver.process_batch([task])
    assert annotation(task, 0)["mfa"]["error"]["code"] == "mfa_model_incomplete"
    assert annotation(task, 1)["mfa"]["error"]["code"] == "mfa_model_unavailable"
    assert runner.calls == []


def test_stage_language_models_yaml_and_bundle_manifest(tmp_path: Path, calibration_dir: Path) -> None:
    (calibration_dir / "mfa_language_models.yaml").write_text(
        "supported:\n  en: {acoustic: my_acoustic, dictionary: my_dict}\n", encoding="utf-8"
    )
    task = AudioTask(data={"segments": [segment(tmp_path, "a", gate("en"))]})
    runner = FakeMFA()
    stage(calibration_dir, runner).process_batch([task])
    assert runner.calls[0][3:5] == ["my_dict", "my_acoustic"]

    # a bundle manifest that does not list an artifact makes it unusable -> REVIEW, not a crash
    (calibration_dir / "manifest.json").write_text(json.dumps({"artifacts": {}}), encoding="utf-8")
    task2 = AudioTask(data={"segments": [segment(tmp_path, "b", gate("en"))]})
    stage(calibration_dir, FakeMFA(), mfa_models={"en": {"acoustic": "a", "dictionary": "d"}}).process_batch([task2])
    assert annotation(task2, 0)["mfa"]["error"]["code"] == "verifier_calibration_unavailable"


def test_stage_rejects_bad_num_jobs() -> None:
    with pytest.raises(ValueError, match="num_jobs"):
        EarlyCutOffMFAResolverStage(num_jobs=0)
    assert EarlyCutOffMFAResolverStage().process_batch([]) == []


def test_unverified_language_model_map_disables_mfa(tmp_path: Path, calibration_dir: Path) -> None:
    (calibration_dir / "mfa_language_models.yaml").write_text("supported:\n  en: {acoustic: a, dictionary: d}\n")
    (calibration_dir / "manifest.json").write_text(json.dumps({"artifacts": {}}), encoding="utf-8")
    task = AudioTask(data={"segments": [segment(tmp_path, "a", gate("en"))]})
    runner = FakeMFA()
    stage(calibration_dir, runner).process_batch([task])
    assert annotation(task, 0)["verifier_status"] == MODEL_UNAVAILABLE_REVIEW
    assert runner.calls == []


def test_candidate_scan_does_not_decode_audio(
    tmp_path: Path, calibration_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nemo_curator.stages.audio.tts.early_cutoff.mfa_resolver as module

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError

    monkeypatch.setattr(module, "load_segment_audio", boom)
    tasks = [AudioTask(data={"segments": [segment(tmp_path, f"s{i}", gate("en"))]}) for i in range(2)]
    grouped, failures = stage(calibration_dir, FakeMFA()).collect_candidates(tasks)
    assert failures == {}
    assert [c.key for c in grouped["en"]] == [(0, 0), (1, 0)]
    assert all(c.duration_sec == 0.0 for c in grouped["en"])


def test_mfa_root_bootstraps_once_then_uses_local_bundle(tmp_path: Path, calibration_dir: Path) -> None:
    root = tmp_path / "mfa_root"
    runner = FakeMFA(mfa_root=root)
    resolver = stage(calibration_dir, runner, mfa_root_dir=str(root))
    for name in ("first", "second"):
        task = AudioTask(data={"segments": [segment(tmp_path, name, gate("en"))]})
        resolver.process_batch([task])
        assert annotation(task, 0)["decision"] == "DROP"
    assert runner.calls[0][3:5] == ["english_us_arpa", "english_us_arpa"]  # bootstrap by model name
    assert Path(runner.calls[1][3]).name == "dictionary.dict"  # then the private bundle copy
    assert Path(runner.calls[1][4]).name == "acoustic"
    assert (root / "model_bundles/en.tar").is_file()
    assert (root / "model_bundles/en.tar.json").is_file()
    assert (root / "alignment_locks/en.lock").is_file()
    local = Path(runner.calls[1][4]).parent
    resolver.teardown()
    assert not local.exists()


def test_model_cache_repairs_stale_lock_state(tmp_path: Path) -> None:
    """A stale ready marker and a partial extraction are repaired."""
    from nemo_curator.stages.audio.tts.early_cutoff.mfa_models import MfaModelCache

    root = tmp_path / "root"
    dictionary = root / "pretrained_models/dictionary/english_us_arpa.dict"
    dictionary.parent.mkdir(parents=True)
    dictionary.write_text("WORD W ER D\n")
    acoustic = root / "extracted_models/acoustic/english_us_arpa_acoustic"
    acoustic.mkdir(parents=True)
    (acoustic / "partial").write_text("broken\n")
    locks = root / "locks"
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> None:
        calls.append(list(command))
        acoustic.mkdir(parents=True, exist_ok=True)
        (acoustic / "final.mdl").write_text("model\n")
        (acoustic / "tree").write_text("tree\n")

    cache = MfaModelCache(root_dir=str(root), lock_dir=str(locks), runner=fake_run)
    command = ["mfa", "align", "/corpus", "english_us_arpa", "english_us_arpa", "/aligned"]
    cache.run(command, "en", tmp_path / "work")
    cache.run(command, "en", tmp_path / "work")
    assert calls[0][3:5] == ["english_us_arpa"] * 2
    assert Path(calls[1][3]).name == "dictionary.dict"
    assert Path(calls[1][4]).name == "acoustic"
    assert calls[1][4] != str(acoustic)
    assert (root / "model_bundles/en.tar").is_file()
    assert not (acoustic / "partial").exists()
    cache.cleanup()


def test_model_cache_bootstrap_incomplete_raises(tmp_path: Path) -> None:
    from nemo_curator.stages.audio.tts.early_cutoff.mfa_models import MfaModelCache

    cache = MfaModelCache(root_dir=str(tmp_path), runner=lambda *_a, **_k: None)
    with pytest.raises(RuntimeError, match="bootstrap incomplete"):
        cache.run(["mfa", "align", "/c", "d", "m", "/o"], "en", tmp_path / "work")


def test_empty_audio_is_materialization_review(tmp_path: Path, calibration_dir: Path) -> None:
    empty = tmp_path / "empty.wav"
    sf.write(empty, np.zeros(0, dtype=np.float32), SR)
    good = segment(tmp_path, "good", gate("en"))
    bad = {"audio_filepath": str(empty), "GranaryV2": {"tn_raw": "hi"}, "early_cut_off_detection": gate("en")}
    task = AudioTask(data={"segments": [bad, good]})
    runner = FakeMFA()
    stage(calibration_dir, runner).process_batch([task])
    assert annotation(task, 0)["mfa"]["error"]["code"] == "mfa_candidate_materialization_failed"
    assert "decoded empty waveform" in annotation(task, 0)["mfa"]["error"]["message"]
    assert annotation(task, 1)["decision"] == "DROP"
    assert len(runner.calls) == 1


def _bundle_ready_root(tmp_path: Path) -> tuple[Path, list[list[str]], object]:
    """MFA root with an already extracted english_us_arpa model and a runner that records calls."""
    root = tmp_path / "root"
    (root / "pretrained_models/dictionary").mkdir(parents=True)
    (root / "pretrained_models/dictionary/english_us_arpa.dict").write_text("WORD W ER D\n")
    acoustic = root / "extracted_models/acoustic/english_us_arpa_acoustic"
    acoustic.mkdir(parents=True)
    (acoustic / "final.mdl").write_text("model\n")
    (acoustic / "tree").write_text("tree\n")
    calls: list[list[str]] = []
    return root, calls, lambda command, **_kwargs: calls.append(list(command))


ALIGN = ["mfa", "align", "/corpus", "english_us_arpa", "english_us_arpa", "/aligned"]


def test_bundle_built_once_and_reused_across_processes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import nemo_curator.stages.audio.tts.early_cutoff.mfa_models as module

    root, calls, runner = _bundle_ready_root(tmp_path)
    created: list[Path] = []
    original = module.MfaModelCache.create_bundle
    monkeypatch.setattr(
        module.MfaModelCache,
        "create_bundle",
        lambda self, command, resolved, bundle: (created.append(bundle), original(self, command, resolved, bundle)),
    )
    exclusive: list[int] = []
    real_flock = module.fcntl.flock
    monkeypatch.setattr(
        module.fcntl,
        "flock",
        lambda fd, op: (exclusive.append(op) if op == module.fcntl.LOCK_EX else None, real_flock(fd, op)),
    )
    first = module.MfaModelCache(root_dir=str(root), runner=runner)
    first.run(ALIGN, "en", tmp_path / "w")
    first.run(ALIGN, "en", tmp_path / "w")
    second = module.MfaModelCache(root_dir=str(root), runner=runner)  # another worker process
    second.run(ALIGN, "en", tmp_path / "w")
    assert len(created) == 1  # extracted model already complete -> bundled once, no MFA bootstrap run
    assert len(exclusive) == 1  # only the build took the exclusive lock
    assert len(calls) == 3
    assert all(Path(call[3]).name == "dictionary.dict" for call in calls)
    assert calls[0][3] == calls[1][3] != calls[2][3]  # per-process private extraction
    first.cleanup()
    second.cleanup()


def test_bundle_rebuilt_when_stale_or_forced(tmp_path: Path) -> None:
    from nemo_curator.stages.audio.tts.early_cutoff.mfa_models import MfaModelCache

    root, _, runner = _bundle_ready_root(tmp_path)

    def build(**kwargs: object) -> int:
        bundle, aligned = MfaModelCache(root_dir=str(root), runner=runner, **kwargs).ensure_bundle(
            ALIGN, "en", tmp_path / "w"
        )
        assert aligned is False  # model already extracted: no MFA bootstrap alignment needed
        return bundle.stat().st_ino

    cache = MfaModelCache(root_dir=str(root), runner=runner)
    built = build()
    bundle = cache.bundle_path("en", tmp_path / "w")
    assert cache.bundle_is_current(ALIGN, bundle)
    assert build() == built  # current -> untouched
    # source model changed -> stale -> rebuilt (os.replace gives a new inode)
    (root / "extracted_models/acoustic/english_us_arpa_acoustic/final.mdl").write_text("new model, longer\n")
    assert not cache.bundle_is_current(ALIGN, bundle)
    rebuilt = build()
    assert rebuilt != built
    assert cache.bundle_is_current(ALIGN, bundle)
    # a different model name or a corrupted sidecar is stale
    assert not cache.bundle_is_current([*ALIGN[:4], "other", ALIGN[5]], bundle)
    bundle.with_name("en.tar.json").write_text("{not json")
    assert not cache.bundle_is_current(ALIGN, bundle)
    repaired = build()
    assert repaired != rebuilt
    # rebuild_bundles forces one rebuild even when current
    assert build(rebuild_bundles=True) != repaired


def test_bundle_checksum_mismatch_is_review(tmp_path: Path, calibration_dir: Path) -> None:
    root, _, runner = _bundle_ready_root(tmp_path)
    from nemo_curator.stages.audio.tts.early_cutoff.mfa_models import MfaModelCache

    bundle, _ = MfaModelCache(root_dir=str(root), runner=runner).ensure_bundle(ALIGN, "en", tmp_path / "w")
    meta = json.loads(bundle.with_name("en.tar.json").read_text())
    bundle.with_name("en.tar.json").write_text(json.dumps({**meta, "sha256": "0" * 64}))
    task = AudioTask(data={"segments": [segment(tmp_path, "a", gate("en"))]})
    stage(calibration_dir, FakeMFA(), mfa_root_dir=str(root)).process_batch([task])
    resolved = annotation(task, 0)
    assert resolved["verifier_status"] == TECHNICAL_ERROR_REVIEW
    assert "checksum mismatch" in resolved["mfa"]["error"]["message"]


def test_num_jobs_defaults_to_stage_cpus(tmp_path: Path, calibration_dir: Path) -> None:
    from nemo_curator.stages.resources import Resources

    assert EarlyCutOffMFAResolverStage().effective_num_jobs == 8
    assert EarlyCutOffMFAResolverStage(num_jobs=3).effective_num_jobs == 3
    resized = EarlyCutOffMFAResolverStage().with_(resources=Resources(cpus=48.0))
    assert resized.effective_num_jobs == 48
    task = AudioTask(data={"segments": [segment(tmp_path, "a", gate("en"))]})
    runner = FakeMFA()
    stage(calibration_dir, runner).process_batch([task])
    assert runner.calls[0][runner.calls[0].index("--num_jobs") + 1] == "8"


def test_sqlite_path_with_uri_special_characters(tmp_path: Path) -> None:
    odd = tmp_path / "we?ird#dir %20"
    odd.mkdir()
    db = odd / "corpus.db"
    create_mfa_db(db, {"utt": 1.0})
    assert sqlite_likelihood_evidence(db, ["utt"])["utt"]["likelihood_status"] == "AVAILABLE"


def test_tar_addressed_candidates(tmp_path: Path, calibration_dir: Path) -> None:
    from tests.stages.audio.tts.test_audio_io import make_tar_segments

    segments = make_tar_segments(tmp_path, [("vid_0_1.wav", 1.0, 22050), ("vid_1_2.wav", 1.5, 22050)])
    for seg in segments:
        meta = seg["GranaryHifi"]["AudioMetadata"]
        meta["tar_key"] = "bucket/prefix/" + Path(meta["tar_key"]).name  # relative key
        seg["early_cut_off_detection"] = gate("en")
    task = AudioTask(data={"orig_sample_id": "vid", "segments": segments})
    runner = FakeMFA()
    stage(
        calibration_dir,
        runner,
        tar_root=str(tmp_path),
        strip_tar_key_prefixes=("bucket/prefix",),
        textgrid_dir=str(tmp_path / "grids"),
    ).process_batch([task])
    corpus_names = sorted(p.name for p in (tmp_path / "grids").iterdir())
    assert corpus_names == ["t0_s0_vid_0_1.TextGrid", "t0_s1_vid_1_2.TextGrid"]
    for index, seconds in ((0, 1.0), (1, 1.5)):
        resolved = annotation(task, index)
        assert resolved["decision"] == "DROP"
        assert resolved["mfa"]["audio_duration_sec"] == pytest.approx(seconds, abs=1e-3)  # resampled to 16 kHz
    # without tar_root the relative key does not resolve -> materialization REVIEW, no MFA run
    task2 = AudioTask(data={"segments": [dict(segments[0], early_cut_off_detection=gate("en"))]})
    runner2 = FakeMFA()
    stage(calibration_dir, runner2).process_batch([task2])
    assert annotation(task2, 0)["mfa"]["error"]["code"] == "mfa_candidate_materialization_failed"
    assert "missing_audio_tar" in annotation(task2, 0)["mfa"]["error"]["message"]
    assert runner2.calls == []
