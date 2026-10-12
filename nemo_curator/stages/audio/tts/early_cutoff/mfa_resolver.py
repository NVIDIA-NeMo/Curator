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

"""CPU early cut-off MFA resolver (step 2 of the endpoint/MFA cascade).

Only segments that ``EarlyCutOffEndpointGateStage`` routed
to verification (``decision=REVIEW``, ``mfa_verification_required=True``) are aligned, with one Montreal Forced
Aligner run per language per batch. The frozen verifier then sets ``DROP`` (MFA-confirmed cut-off) or ``OK``;
anything MFA cannot settle (unsupported language, no model, technical failure) stays ``REVIEW``. Rows and segments
are never dropped; every gate annotation is re-stamped with ``cascade_step="mfa_resolver"``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.stages.audio.tts.audio_io import (
    SegmentAudioError,
    iter_segments,
    load_segment_audio,
    make_audio_config,
)
from nemo_curator.stages.audio.tts.early_cutoff.endpoint_gate import (
    _TEXT_FALLBACKS,
    CASCADE_VERSION,
    SCHEMA_VERSION,
    SUPPORTED_LANGUAGES,
    _error,
    _verify_bundle_artifact,
    language_code,
)
from nemo_curator.stages.audio.tts.early_cutoff.mfa_models import MfaModelCache
from nemo_curator.stages.audio.tts.early_cutoff.mfa_runtime import (
    build_align_command,
    discover_database,
    discover_textgrids,
    extract_textgrid_evidence,
    preserve_database,
    run_alignment,
    safe_name,
    sqlite_likelihood_evidence,
    write_corpus_utterance,
)
from nemo_curator.stages.audio.tts.early_cutoff.verifier import verifier_components
from nemo_curator.stages.audio.tts.fields import MISSING, get_dotted, set_dotted
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

MODEL_UNAVAILABLE_REVIEW = "MFA_MODEL_UNAVAILABLE_REVIEW"
TECHNICAL_ERROR_REVIEW = "MFA_TECHNICAL_ERROR_REVIEW"
UNSUPPORTED_LANGUAGE_REVIEW = "MFA_UNSUPPORTED_LANGUAGE_REVIEW"
_VERIFICATION_STATUS = {
    MODEL_UNAVAILABLE_REVIEW: "MFA_MODEL_UNAVAILABLE",
    TECHNICAL_ERROR_REVIEW: "MFA_TECHNICAL_ERROR",
    UNSUPPORTED_LANGUAGE_REVIEW: "UNSUPPORTED_LANGUAGE",
}

# Default language -> model map (MFA 3.4.1 pretrained models); ``language_models`` / ``mfa_models`` override it.
DEFAULT_MFA_MODELS: dict[str, dict[str, str]] = {
    "bg": {"acoustic": "bulgarian_mfa", "dictionary": "bulgarian_mfa"},
    "cs": {"acoustic": "czech_mfa", "dictionary": "czech_mfa"},
    "de": {"acoustic": "german_mfa", "dictionary": "german_mfa"},
    "el": {"acoustic": "greek_cv", "dictionary": "greek_cv"},
    "en": {"acoustic": "english_us_arpa", "dictionary": "english_us_arpa"},
    "es": {"acoustic": "spanish_mfa", "dictionary": "spanish_latin_america_mfa"},
    "fr": {"acoustic": "french_mfa", "dictionary": "french_mfa"},
    "hr": {"acoustic": "serbocroatian_mfa", "dictionary": "serbocroatian_mfa"},
    "hu": {"acoustic": "hungarian_cv", "dictionary": "hungarian_cv"},
    "it": {"acoustic": "italian_cv", "dictionary": "italian_cv"},
    "nl": {"acoustic": "dutch_cv", "dictionary": "dutch_cv"},
    "pl": {"acoustic": "polish_mfa", "dictionary": "polish_mfa"},
    "pt": {"acoustic": "portuguese_mfa", "dictionary": "portuguese_brazil_mfa"},
    "ro": {"acoustic": "romanian_cv", "dictionary": "romanian_cv"},
    "ru": {"acoustic": "russian_mfa", "dictionary": "russian_mfa"},
    "sv": {"acoustic": "swedish_mfa", "dictionary": "swedish_mfa"},
    "uk": {"acoustic": "ukrainian_mfa", "dictionary": "ukrainian_mfa"},
}


def review(code: str, message: str | None = None, *, status: str) -> dict[str, Any]:
    """Resolution for a candidate MFA could not settle."""
    return {
        "decision": "REVIEW",
        "is_suspicious": True,
        "verifier_status": status,
        "verification_status": _VERIFICATION_STATUS.get(status, status),
        "verifier": None,
        "mfa": {
            "alignment_status": "ERROR",
            "likelihood_status": "MODEL_UNAVAILABLE" if status == MODEL_UNAVAILABLE_REVIEW else "UNAVAILABLE",
            "error": _error(code, message),
        },
    }


def _is_versioned(annotation: Mapping[str, Any]) -> bool:
    return annotation.get("schema_version") == SCHEMA_VERSION and annotation.get("cascade_version") == CASCADE_VERSION


def is_resolved(annotation: Any) -> bool:  # noqa: ANN401
    """Final resolver output (idempotent skip)."""
    return (
        isinstance(annotation, dict)
        and _is_versioned(annotation)
        and annotation.get("cascade_step") == "mfa_resolver"
        and annotation.get("decision") in {"OK", "DROP"}
    )


def is_candidate(annotation: Any) -> bool:  # noqa: ANN401
    """Gate output routed to MFA verification."""
    return (
        isinstance(annotation, dict)
        and _is_versioned(annotation)
        and annotation.get("mfa_verification_required") is True
        and annotation.get("decision") == "REVIEW"
    )


def calibration_thresholds(calibration: Mapping[str, Any], language: str) -> Mapping[str, Any]:
    """Verifier thresholds for ``language`` (global thresholds with per-language overrides)."""
    if "global_thresholds" in calibration:
        return (calibration.get("language_overrides") or {}).get(language, calibration["global_thresholds"])
    return calibration["thresholds"]


def resolve_candidate(
    gate: Mapping[str, Any],
    mfa: dict[str, Any],
    null_evidence: Mapping[str, Sequence[float]],
    threshold: float,
) -> dict[str, Any]:
    """DROP/OK decision for an aligned candidate; REVIEW if MFA likelihoods are unavailable."""
    if mfa.get("likelihood_status") != "AVAILABLE":
        resolution = review("mfa_likelihood_unavailable", status=TECHNICAL_ERROR_REVIEW)
        resolution["mfa"] = mfa
        return resolution
    components = verifier_components({"prediction": gate, "mfa": mfa}, null_evidence)
    confirmed = float(components["score"]) > threshold
    return {
        "decision": "DROP" if confirmed else "OK",
        "is_suspicious": confirmed,
        "verifier_status": "AVAILABLE",
        "verification_status": "AVAILABLE",
        "verifier": components,
        "verifier_threshold": threshold,
        "mfa": mfa,
    }


@dataclass
class _Candidate:
    """A gate candidate; audio is decoded only when the language's MFA corpus is written."""

    key: tuple[int, int]
    mfa_id: str
    text: str
    segment: dict[str, Any]
    gate: dict[str, Any]
    duration_sec: float = 0.0


@dataclass
class EarlyCutOffMFAResolverStage(ProcessingStage[AudioTask, AudioTask]):
    """Resolve early cut-off gate candidates with Montreal Forced Aligner and the frozen verifier.

    Requires the MFA 3.x CLI (``mfa_command``; ``mfa_root_dir`` sets ``MFA_ROOT_DIR`` and prepends
    ``<root>/runtime/bin`` to ``PATH`` for the subprocess) with the pretrained models of ``DEFAULT_MFA_MODELS``
    downloaded, and the verifier calibration of the endpoint bundle (``bundle_dir`` fills
    ``english_calibration``, ``multilingual_calibration``, ``multilingual_thresholds``, ``language_models`` and
    ``bundle_manifest``). Missing pieces never fail the stage: affected candidates stay ``REVIEW``.

    With ``mfa_root_dir`` set (and no ``command_template``), models are aligned from immutable per-language bundles
    (``model_bundle_dir``, default ``<mfa_root_dir>/model_bundles``) built once under an exclusive lock in
    ``alignment_lock_dir`` (default ``<mfa_root_dir>/alignment_locks``); ``rebuild_model_bundles`` forces one
    rebuild per process. ``num_jobs`` defaults to ``int(resources.cpus)``.

    MFA's ``--single_speaker`` adaptation is estimated over all candidates of one language in one
    ``process_batch`` call, so alignments (and which hard utterances MFA fails to align -> ``REVIEW``) depend on the
    batch: keep ``batch_size`` large.
    """

    text_key: str = "tn_raw"
    output_key: str = "early_cut_off_detection"
    bundle_dir: str | None = None
    english_calibration: str | None = None
    multilingual_calibration: str | None = None
    multilingual_thresholds: str | None = None
    language_models: str | None = None
    bundle_manifest: str | None = None
    mfa_models: dict[str, dict[str, str]] | None = None
    mfa_command: str = "mfa"
    mfa_root_dir: str | None = None
    command_template: list[str] | None = None
    num_jobs: int | None = None
    clean_temp: bool = True
    extra_args: tuple[str, ...] = ("--single_speaker",)
    timeout_sec: float | None = 14400.0
    eof_threshold_sec: float = 0.03
    temporary_root: str | None = None
    alignment_lock_dir: str | None = None
    model_bundle_dir: str | None = None
    rebuild_model_bundles: bool = False
    textgrid_dir: str | None = None
    preserve_database_dir: str | None = None
    sample_rate: int = 16000
    tar_root: str | None = None
    strip_tar_key_prefixes: tuple[str, ...] = ()
    audio_source: str = "original"
    enhanced_key: str = "enhanced_audio"
    enhanced_fallback: bool = False
    overwrite: bool = False
    command_runner: Callable[..., Any] | None = None
    name: str = "EarlyCutOffMFAResolver"
    resources: Resources = field(default_factory=lambda: Resources(cpus=8.0))
    batch_size: int = 64

    def __post_init__(self) -> None:
        super().__init__()
        if self.bundle_dir:
            root = Path(self.bundle_dir).expanduser()
            self.english_calibration = self.english_calibration or str(root / "english_verifier.json")
            self.multilingual_calibration = self.multilingual_calibration or str(root / "multilingual_verifier.json")
            self.multilingual_thresholds = self.multilingual_thresholds or str(root / "multilingual_thresholds.json")
            self.language_models = self.language_models or str(root / "mfa_language_models.yaml")
            self.bundle_manifest = self.bundle_manifest or str(root / "manifest.json")
        if self.num_jobs is not None and self.num_jobs < 1:
            msg = "num_jobs must be positive"
            raise ValueError(msg)
        self._models: dict[str, Any] | None = None
        self._calibrations: dict[str, dict[str, Any]] = {}
        self._model_cache: MfaModelCache | None = None

    def teardown(self) -> None:
        if self._model_cache is not None:
            self._model_cache.cleanup()

    @property
    def effective_num_jobs(self) -> int:
        """``num_jobs`` if set, else one MFA job per CPU reserved for the stage (``resources.cpus``)."""
        if self.num_jobs is not None:
            return self.num_jobs
        return max(1, int(self.resources.cpus))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_key.split(".")[0]]

    # ------------------------------------------------------------------ configuration

    def _manifest_path(self) -> Path | None:
        return Path(self.bundle_manifest).expanduser() if self.bundle_manifest else None

    def _read_artifact(self, path_value: str) -> str:
        path = Path(path_value).expanduser()
        _verify_bundle_artifact(self._manifest_path(), path)
        return path.read_text(encoding="utf-8")

    def model_for_language(self, language: str) -> Mapping[str, Any] | None:
        if self._models is None:
            if self.mfa_models is not None:
                self._models = dict(self.mfa_models)
            elif self.language_models and Path(self.language_models).expanduser().is_file():
                self._models = self._load_language_models(self.language_models)
            else:
                self._models = dict(DEFAULT_MFA_MODELS)
        model = self._models.get(language)
        return model if isinstance(model, dict) else None

    def _load_language_models(self, path: str) -> dict[str, Any]:
        """Model map from the bundle YAML; an unreadable/unverified map disables MFA (candidates stay REVIEW)."""
        import yaml

        try:
            payload = yaml.safe_load(self._read_artifact(path)) or {}
        except Exception:  # noqa: BLE001
            logger.exception("Cannot load MFA language model map {}", path)
            return {}
        return dict(payload.get("supported") or {})

    def load_calibration(self, language: str) -> dict[str, Any]:
        family = "en" if language == "en" else "multi"
        if family in self._calibrations:
            return self._calibrations[family]
        path = self.english_calibration if family == "en" else self.multilingual_calibration
        if not path:
            msg = "MFA verifier calibration is not configured"
            raise ValueError(msg)
        calibration = json.loads(self._read_artifact(path))
        if family == "multi" and self.multilingual_thresholds:
            thresholds = json.loads(self._read_artifact(self.multilingual_thresholds))
            calibration["global_thresholds"] = thresholds["global_thresholds"]
            calibration["language_overrides"] = thresholds.get("language_overrides", {})
        if not isinstance(calibration.get("null_evidence"), dict):
            msg = f"verifier calibration {path} has no null_evidence"
            raise TypeError(msg)
        self._calibrations[family] = calibration
        return calibration

    def _subprocess_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.mfa_root_dir:
            root = Path(self.mfa_root_dir).expanduser()
            env["MFA_ROOT_DIR"] = str(root)
            env["PATH"] = str(root / "runtime" / "bin") + os.pathsep + env.get("PATH", "")
        return env

    def _executable_available(self, env: Mapping[str, str]) -> bool:
        if self.command_runner is not None:
            return True
        return shutil.which(self.mfa_command, path=env.get("PATH")) is not None or Path(self.mfa_command).is_file()

    # ------------------------------------------------------------------ candidates

    def _resolve_text(self, segment: dict[str, Any]) -> Any:  # noqa: ANN401
        for key in (self.text_key, *_TEXT_FALLBACKS):
            value = get_dotted(segment, key, MISSING)
            if value is not MISSING:
                return value
        return MISSING

    def _make_candidate(self, key: tuple[int, int], segment: dict[str, Any], gate: dict[str, Any]) -> _Candidate:
        """Validate a candidate without decoding audio."""
        text = self._resolve_text(segment)
        if not isinstance(text, str) or not text.strip():
            msg = "candidate transcript is empty"
            raise ValueError(msg)
        member = get_dotted(segment, "GranaryHifi.AudioMetadata.tar_member") or segment.get("audio_filepath")
        if not member:
            msg = "candidate audio metadata is incomplete"
            raise ValueError(msg)
        stem = Path(str(member)).stem
        return _Candidate(key, safe_name(f"t{key[0]}_s{key[1]}_{stem}"), text.strip(), segment, gate)

    def collect_candidates(
        self, tasks: Sequence[AudioTask]
    ) -> tuple[dict[str, list[_Candidate]], dict[tuple[int, int], dict[str, Any]]]:
        """Group gate candidates by language; return early resolutions for those MFA cannot handle."""
        grouped: dict[str, list[_Candidate]] = {}
        failures: dict[tuple[int, int], dict[str, Any]] = {}
        for task_index, task in enumerate(tasks):
            for segment_index, segment in iter_segments(task.data):
                gate = get_dotted(segment, self.output_key)
                rerun = self.overwrite and is_resolved(gate) and gate.get("mfa_verification_required") is True
                if not (is_candidate(gate) or rerun):
                    continue
                key = (task_index, segment_index)
                language = language_code(str(gate.get("language", "")))
                if language == "mt" or language not in SUPPORTED_LANGUAGES:
                    failures[key] = review(
                        "unsupported_language",
                        f"MFA resolver does not support {language or 'unknown'}",
                        status=UNSUPPORTED_LANGUAGE_REVIEW,
                    )
                    continue
                if self.model_for_language(language) is None:
                    failures[key] = review(
                        "mfa_model_unavailable",
                        f"no MFA model configured for {language}",
                        status=MODEL_UNAVAILABLE_REVIEW,
                    )
                    continue
                try:
                    candidate = self._make_candidate(key, segment, gate)
                except ValueError as exc:
                    failures[key] = review(
                        "mfa_candidate_materialization_failed",
                        f"{type(exc).__name__}: {exc}",
                        status=TECHNICAL_ERROR_REVIEW,
                    )
                else:
                    grouped.setdefault(language, []).append(candidate)
        return grouped, failures

    # ------------------------------------------------------------------ alignment

    def materialize(
        self, candidates: Sequence[_Candidate], corpus: Path
    ) -> tuple[dict[str, _Candidate], dict[tuple[int, int], dict[str, Any]]]:
        """Decode candidates one at a time into the MFA corpus; returns ``{corpus_key: candidate}`` + failures."""
        audio_config = make_audio_config(self)
        index: dict[str, _Candidate] = {}
        used: set[str] = set()
        failures: dict[tuple[int, int], dict[str, Any]] = {}
        for ordinal, candidate in enumerate(candidates):
            try:
                waveform, _ = load_segment_audio(candidate.segment, self.sample_rate, audio_config)
                if waveform.shape[-1] <= 0:
                    msg = "decoded empty waveform"
                    raise ValueError(msg)  # noqa: TRY301
                key = candidate.mfa_id
                while key.casefold() in used:
                    key = f"{key}_{ordinal}"
                write_corpus_utterance(corpus, key, waveform, candidate.text, sample_rate=self.sample_rate)
            except (SegmentAudioError, ValueError, OSError, RuntimeError) as exc:
                failures[candidate.key] = review(
                    "mfa_candidate_materialization_failed",
                    f"{type(exc).__name__}: {exc}",
                    status=TECHNICAL_ERROR_REVIEW,
                )
                continue
            candidate.duration_sec = waveform.shape[-1] / self.sample_rate
            used.add(key.casefold())
            index[key] = candidate
        return index, failures

    def _run_alignment(self, command: list[str], language: str, root: Path, env: Mapping[str, str]) -> None:
        runner = self.command_runner or subprocess.run
        if not self.mfa_root_dir or self.command_template:
            # No shared MFA_ROOT_DIR to protect (or a custom command layout): run as given.
            run_alignment(command, runner=runner, timeout_sec=self.timeout_sec, env=env)
            return
        if self._model_cache is None:
            self._model_cache = MfaModelCache(
                root_dir=self.mfa_root_dir,
                lock_dir=self.alignment_lock_dir,
                bundle_dir=self.model_bundle_dir,
                rebuild_bundles=self.rebuild_model_bundles,
                runner=runner,
                timeout_sec=self.timeout_sec,
                env=env,
            )
        self._model_cache.run(command, language, root)

    def align_and_resolve(  # noqa: PLR0913
        self,
        language: str,
        index: Mapping[str, _Candidate],
        root: Path,
        model: Mapping[str, Any],
        calibration: Mapping[str, Any],
        env: Mapping[str, str],
    ) -> dict[tuple[int, int], dict[str, Any]]:
        """One ``mfa align`` over the materialized corpus in ``root/corpus``; DROP/OK/REVIEW per candidate."""
        corpus, aligned, work = root / "corpus", root / "aligned", root / "work"
        dictionary = str(model.get("dictionary"))
        acoustic = str(model.get("acoustic_model") or model.get("acoustic") or model.get("model"))
        command = build_align_command(
            self.mfa_command,
            corpus,
            dictionary,
            acoustic,
            aligned,
            work,
            num_jobs=self.effective_num_jobs,
            clean=self.clean_temp,
            extra_args=tuple(self.extra_args),
            command_template=self.command_template,
        )
        self._run_alignment(command, language, root, env)
        grids = discover_textgrids(aligned)
        database = discover_database(work)
        if database is not None and self.preserve_database_dir:
            database = preserve_database(
                database, Path(self.preserve_database_dir).expanduser() / f"{root.name}.{language}.mfa.db"
            )
        likelihoods = sqlite_likelihood_evidence(database, index) if database is not None else {}
        threshold = float(calibration_thresholds(calibration, language)["verifier_threshold"])
        null_evidence = calibration["null_evidence"]
        resolutions: dict[tuple[int, int], dict[str, Any]] = {}
        for utterance_id, candidate in index.items():
            grid = grids.get(utterance_id.casefold())
            if grid is None:
                resolutions[candidate.key] = review(
                    "mfa_textgrid_missing", f"TextGrid not found for {utterance_id}", status=TECHNICAL_ERROR_REVIEW
                )
                continue
            try:
                mfa = extract_textgrid_evidence(
                    grid, audio_duration_sec=candidate.duration_sec, eof_threshold_sec=self.eof_threshold_sec
                )
                if self.textgrid_dir:
                    persistent = Path(self.textgrid_dir).expanduser() / f"{utterance_id}.TextGrid"
                    persistent.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(grid, persistent)
                    mfa["textgrid_path"] = str(persistent)
                else:
                    mfa.pop("textgrid_path", None)
                mfa.update(likelihoods.get(utterance_id.casefold(), {"likelihood_status": "UNAVAILABLE"}))
                resolutions[candidate.key] = resolve_candidate(candidate.gate, mfa, null_evidence, threshold)
            except Exception as exc:  # noqa: BLE001
                logger.exception("MFA candidate resolution failed for {}", utterance_id)
                resolutions[candidate.key] = review(
                    "mfa_candidate_resolution_failed", f"{type(exc).__name__}: {exc}", status=TECHNICAL_ERROR_REVIEW
                )
        return resolutions

    def resolve_language(
        self, language: str, candidates: Sequence[_Candidate], env: Mapping[str, str]
    ) -> dict[tuple[int, int], dict[str, Any]]:
        try:
            calibration = self.load_calibration(language)
        except Exception as exc:  # noqa: BLE001
            return {
                c.key: review(
                    "verifier_calibration_unavailable", f"{type(exc).__name__}: {exc}", status=TECHNICAL_ERROR_REVIEW
                )
                for c in candidates
            }
        model = self.model_for_language(language) or {}
        if not model.get("dictionary") or not (
            model.get("acoustic_model") or model.get("acoustic") or model.get("model")
        ):
            return {
                c.key: review(
                    "mfa_model_incomplete",
                    f"{language} requires dictionary and acoustic_model",
                    status=MODEL_UNAVAILABLE_REVIEW,
                )
                for c in candidates
            }
        resolutions: dict[tuple[int, int], dict[str, Any]] = {}
        try:
            with tempfile.TemporaryDirectory(prefix=f"truncation_mfa_{language}_", dir=self.temporary_root) as tmp:
                root = Path(tmp)
                index, failures = self.materialize(candidates, root / "corpus")
                resolutions.update(failures)
                if index:
                    resolutions.update(self.align_and_resolve(language, index, root, model, calibration, env))
        except Exception as exc:  # noqa: BLE001
            logger.exception("MFA resolution failed for {}", language)
            for candidate in candidates:
                resolutions.setdefault(
                    candidate.key,
                    review("mfa_runtime_failed", f"{type(exc).__name__}: {exc}", status=TECHNICAL_ERROR_REVIEW),
                )
        return resolutions

    # ------------------------------------------------------------------ stage API

    def resolve(self, tasks: Sequence[AudioTask]) -> dict[tuple[int, int], dict[str, Any]]:
        grouped, resolutions = self.collect_candidates(tasks)
        if not grouped:
            return resolutions
        env = self._subprocess_env()
        if not self._executable_available(env):
            for candidates in grouped.values():
                for candidate in candidates:
                    resolutions[candidate.key] = review(
                        "mfa_executable_unavailable", self.mfa_command, status=MODEL_UNAVAILABLE_REVIEW
                    )
            return resolutions
        for language in sorted(grouped):
            candidates = sorted(grouped[language], key=lambda c: c.key)
            logger.info("Resolving {} early cut-off candidates for {} with MFA", len(candidates), language)
            resolutions.update(self.resolve_language(language, candidates, env))
        return resolutions

    def apply(self, tasks: Sequence[AudioTask], resolutions: Mapping[tuple[int, int], dict[str, Any]]) -> None:
        for task_index, task in enumerate(tasks):
            for segment_index, segment in iter_segments(task.data):
                annotation = get_dotted(segment, self.output_key)
                if not isinstance(annotation, dict) or (is_resolved(annotation) and not self.overwrite):
                    continue
                updated = dict(annotation)
                resolution = resolutions.get((task_index, segment_index))
                if resolution is not None:
                    updated.update(resolution)
                elif annotation.get("mfa_verification_required") is True:
                    updated.update(review("mfa_resolution_missing", status=TECHNICAL_ERROR_REVIEW))
                updated.update(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "cascade_version": CASCADE_VERSION,
                        "cascade_step": "mfa_resolver",
                        "annotation_only": True,
                    }
                )
                set_dotted(segment, self.output_key, updated)

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        tasks = list(tasks)
        self.apply(tasks, self.resolve(tasks))
        return tasks
