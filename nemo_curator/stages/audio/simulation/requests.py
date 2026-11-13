# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Meeting requests on the existing manifest -> AudioTask protocol.

One manifest row requests exactly one complete mixed meeting. Planning expands
native YAML counts without generating audio. Execution groups compatible rows
and calls the native bulk recipe once per group, capped at the selected count.
All acoustic settings remain native; count, output/cache paths, CPU allocation
and reference-export flags are the documented adapter-owned execution settings.

The initial adapter supports local, already aligned embedded MonoCuts without
lazy transforms, noise, RIR generation, forced-alignment splitting or transition fitting. These
costly modes remain available through the separate native YAML API. Inputs are
bounded to 10,000 files/1 GiB and the native duration target to 300 seconds; native execution
has the existing 3600-second timeout. These bounds include preparation cost, not
only the number of emitted meetings. Constructors never generate or write files.

Every supplied cut must have matching declared sampling rate and exactly one
full-cut supervision with a nonempty transcript, usable word timings and a
nonempty whitespace-free string speaker ID. These are metadata checks, not
inspection of source waveform samples. Pre-resample and align source speech;
the pinned native mixer does not resample mismatched inputs.
"""

import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib.metadata import distribution, version
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from lhotse import MonoCut

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, Role, StageContract, StaticHints
from nemo_curator.stages.audio.common import ManifestReaderStage
from nemo_curator.stages.audio.simulation._runner import validate_config
from nemo_curator.stages.audio.simulation.fastmss import FastMSSManifestReaderStage, FastMSSSimulationStage
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask, EmptyTask, FileGroupTask

_MAX_FILES = 10_000
_MAX_BYTES = 1024**3
_MAX_REQUESTS = 100_000
_MAX_DURATION = 300
_MAX_BATCH = 32
_MAX_TIMEOUT = 3600
# Leave room for the UUID, separator and extension in a 255-byte filename component.
_MAX_EXPORTED_SPEAKER_BYTES = 200


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _worker_identity() -> dict:
    packages = {
        name: version(name)
        for name in (
            "fastmss",
            "hydra-core",
            "omegaconf",
            "lhotse",
            "numpy",
            "soundfile",
            "scipy",
            "pyroomacoustics",
            "tqdm",
        )
    }
    return {
        "python": sys.version,
        "packages": packages,
        "fastmss_source": distribution("fastmss").read_text("direct_url.json"),
    }


def _eligible_speaker(cut: "MonoCut", cfg: dict) -> str | None:
    supervision = cut.supervisions[0]
    excluded = {str(s) for s in cfg.get("exclude_speakers") or []}
    if supervision.speaker in excluded:
        return None
    if not cfg.get("min_utt_duration", 0) <= cut.duration <= cfg.get("max_utt_duration", 20):
        return None
    return supervision.speaker


def _finite_seconds(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _admit_export_speaker(speaker: str, enabled: bool) -> None:
    if enabled and (
        any(character in speaker for character in ("/", "\\", "\x00"))
        or len(speaker.encode("utf-8")) > _MAX_EXPORTED_SPEAKER_BYTES
    ):
        msg = "save_spk requires filename-compatible speaker IDs: no path separators/NUL and at most 200 UTF-8 bytes"
        raise ValueError(msg)


def _admit_supervision(cut: "MonoCut", samplerate: int) -> None:
    supervisions = cut.supervisions
    if len(supervisions) != 1:
        msg = "Every native cut must have exactly one supervision; split or materialize multi-annotation speech first"
        raise ValueError(msg)
    supervision = supervisions[0]
    speaker = supervision.speaker
    if not isinstance(speaker, str) or not speaker or any(character.isspace() for character in speaker):
        msg = "Every supervision requires a nonempty string speaker ID without whitespace for RTTM"
        raise ValueError(msg)
    if not isinstance(supervision.text, str) or not supervision.text.strip():
        msg = "Prepared speech requires nonempty supervision reference text"
        raise ValueError(msg)
    duration, start, length = cut.duration, supervision.start, supervision.duration
    tolerance = 1 / samplerate
    if (
        not all(_finite_seconds(value) for value in (duration, start, length))
        or duration <= 0
        or length <= 0
        or not 0 <= start <= tolerance
        or abs(length - duration) > tolerance
    ):
        msg = "The single supervision must cover the complete cut, within one sample"
        raise ValueError(msg)
    words = (supervision.alignment or {}).get("word")
    if not isinstance(words, list) or not words:
        msg = "Every supervision requires nonempty word alignment"
        raise ValueError(msg)
    previous_end = 0
    for word in words:
        start, length, symbol = word.start, word.duration, word.symbol
        if (
            not all(_finite_seconds(value) for value in (start, length))
            or start < 0
            or length <= 0
            or start < previous_end - tolerance
            or start + length > duration + tolerance
            or not isinstance(symbol, str)
            or not symbol.strip()
        ):
            msg = "Word alignment must be finite, ordered and inside the complete cut"
            raise ValueError(msg)
        previous_end = start + length


@dataclass(kw_only=True)
class _NativeInputs:
    native_inputs_path: str
    engine_path: str
    recipe_path: str
    worker_lock_path: str

    def _check_roots(self) -> None:
        for name in ("native_inputs_path", "engine_path", "recipe_path"):
            path = Path(getattr(self, name)).expanduser().resolve()
            if not path.is_dir():
                msg = f"{name} must be an existing local directory: {path}"
                raise ValueError(msg)
            setattr(self, name, str(path))
        if (
            Path(self.engine_path).name != "fastmss"
            or Path(self.recipe_path).name != "recipes"
            or Path(self.engine_path).parent != Path(self.recipe_path).parent
            or not (Path(self.recipe_path) / "sim.py").is_file()
        ):
            msg = "Requests require sibling fastmss/ and standard recipes/sim.py trees"
            raise ValueError(msg)

    def _write_worker_lock(self, destination: Path) -> None:
        lock = Path(self.worker_lock_path).expanduser().resolve()
        if lock == destination or any(
            lock.is_relative_to(Path(root)) for root in (self.native_inputs_path, self.engine_path, self.recipe_path)
        ):
            msg = "Worker lock must be separate from the manifest and immutable native input trees"
            raise ValueError(msg)
        expected_worker = _worker_identity()
        if lock.exists():
            if json.loads(lock.read_text()) != expected_worker:
                msg = "Worker lock belongs to a different environment; choose a new planning location"
                raise ValueError(msg)
        else:
            lock.parent.mkdir(parents=True, exist_ok=True)
            with lock.open("x", encoding="utf-8") as stream:
                json.dump(expected_worker, stream, sort_keys=True)

    def _check_worker(self) -> None:
        lock = Path(self.worker_lock_path).expanduser().resolve()
        if not lock.is_file() or json.loads(lock.read_text()) != _worker_identity():
            msg = "Worker environment changed or is unbound; prepare a new worker lock and request manifest"
            raise ValueError(msg)
        self.worker_lock_path = str(lock)

    def _input(self, value: str, *, config: bool = False) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(self.recipe_path).parent / path
        path = path.resolve()
        roots = [Path(self.native_inputs_path), *([Path(self.recipe_path)] if config else [])]
        if not path.exists() or not any(path.is_relative_to(root) for root in roots):
            msg = f"Native dependency missing or outside declared input trees: {path}"
            raise ValueError(msg)
        return path

    def _identity(self) -> str:
        """Bind every declared input byte; enforce admission cost before reading it."""
        digest = hashlib.sha256()
        count = size = 0
        deadline = time.monotonic() + 60
        for name in ("native_inputs_path", "engine_path", "recipe_path"):
            root = Path(getattr(self, name))
            digest.update(str(root).encode())
            for current, directories, files in os.walk(root):
                directories.sort()
                for entry in [*directories, *files]:
                    if (Path(current) / entry).is_symlink():
                        msg = "Symlinked native trees are unsupported by meeting requests"
                        raise ValueError(msg)
                for filename in sorted(files):
                    path = Path(current) / filename
                    count += 1
                    size += path.stat().st_size
                    if count > _MAX_FILES or size > _MAX_BYTES or time.monotonic() > deadline:
                        msg = "Native inputs exceed the 10,000-file/1-GiB/60-second admission budget"
                        raise ValueError(msg)
                    digest.update(str(path.relative_to(root)).encode() + b"\0")
                    with path.open("rb") as stream:
                        while chunk := stream.read(1024 * 1024):
                            if time.monotonic() > deadline:
                                msg = "Native input hashing exceeds the 60-second admission budget"
                                raise ValueError(msg)
                            digest.update(chunk)
        digest.update(json.dumps(_worker_identity(), sort_keys=True).encode())
        return digest.hexdigest()

    def _plan(self, config_path: str, overrides: list[str]) -> dict:
        """Compose native defaults and admit the supported scope without audio I/O."""
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        config_path = self._input(config_path)
        if config_path.suffix != ".yaml":
            msg = "Request configuration must be a local .yaml file"
            raise ValueError(msg)
        # Reuse native argument validation, including adapter-owned Hydra settings.
        FastMSSSimulationStage(output_dir="/unused", overrides=overrides)
        previous = os.environ.get("FASTMSS_ROOT")
        os.environ["FASTMSS_ROOT"] = str(Path(self.recipe_path).parent)
        try:
            with initialize_config_dir(config_dir=str(config_path.parent), version_base=None):
                composed = compose(config_name=config_path.stem, overrides=overrides, return_hydra_config=True)
                for source in composed.hydra.runtime.config_sources:
                    if source.schema == "file":
                        self._input(source.path, config=True)
                    elif source.schema == "pkg" and source.provider != "hydra":
                        msg = "External package config defaults are unsupported"
                        raise ValueError(msg)
                OmegaConf.set_struct(composed, False)
                del composed["hydra"]
                raw = json.dumps(OmegaConf.to_container(composed, resolve=False))
                if re.search(r"\$\{[^{}]*:", raw.replace("${oc.env:FASTMSS_ROOT}", "")):
                    msg = "Dynamic resolvers other than FASTMSS_ROOT are unsupported"
                    raise ValueError(msg)
                cfg = OmegaConf.to_container(composed, resolve=True)
        finally:
            if previous is None:
                os.environ.pop("FASTMSS_ROOT", None)
            else:
                os.environ["FASTMSS_ROOT"] = previous
        count, duration = cfg.get("n_meetings"), cfg.get("duration")
        if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= _MAX_REQUESTS:
            msg = "n_meetings must be a positive integer at most 100,000"
            raise ValueError(msg)
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or not 0 < duration <= _MAX_DURATION
        ):
            msg = "Native duration target must be finite, positive and at most 300 seconds"
            raise ValueError(msg)
        validate_config(cfg)
        if (
            cfg.get("add_noise")
            or cfg.get("reverberate")
            or cfg.get("hmm_fit_transitions_to")
            or (cfg.get("split_fa_factor") or 0) > 0
        ):
            msg = "Pre-stage aligned speech: noise, RIR generation, fitting and FA splitting are unsupported by this adapter"
            raise ValueError(msg)
        self._admit_speech(cfg)
        return {"config_path": str(config_path), "overrides": overrides, "native_config": cfg}

    def _admit_cut(self, cut: dict, samplerate: int) -> "MonoCut":
        if cut.get("type") != "MonoCut" or not cut.get("recording", {}).get("sources"):
            msg = "Meeting requests require embedded file-backed MonoCuts"
            raise ValueError(msg)
        if cut.get("transforms") or cut["recording"].get("transforms"):
            msg = "Lazy Lhotse transforms are unsupported; materialize transformed speech as local WAVs first"
            raise ValueError(msg)
        declared_rate = cut["recording"].get("sampling_rate")
        if isinstance(declared_rate, bool) or not isinstance(declared_rate, int) or declared_rate != samplerate:
            msg = "Recording sampling_rate must match native samplerate; resample speech with consistent alignments first"
            raise ValueError(msg)
        for source in cut["recording"]["sources"]:
            if source.get("type") != "file":
                msg = "Remote or command recording sources are unsupported"
                raise ValueError(msg)
            self._input(source["source"])
        from lhotse import CutSet
        from lhotse import validate as validate_lhotse

        try:
            decoded = next(iter(CutSet.from_dicts([cut])))
            validate_lhotse(decoded, read_data=False)
        except (AssertionError, AttributeError, KeyError, TypeError, ValueError) as error:
            msg = "Invalid Lhotse speech cut or supervision/alignment metadata"
            raise ValueError(msg) from error
        _admit_supervision(decoded, samplerate)
        return decoded

    def _admit_speech(self, cfg: dict) -> None:
        samplerate = cfg.get("samplerate")
        if isinstance(samplerate, bool) or not isinstance(samplerate, int) or samplerate <= 0:
            msg = "Native samplerate must be a positive integer"
            raise ValueError(msg)
        prefix = cfg["manifest_prefix"]
        if not isinstance(prefix, str) or not prefix or "/" in prefix or "\\" in prefix or prefix in {".", ".."}:
            msg = "manifest_prefix must be a plain name without path components"
            raise ValueError(msg)
        speakers = Counter()
        seen_cut_ids = set()
        rows = size = 0
        deadline = time.monotonic() + 60
        for split in cfg["dset_splits"]:
            manifest = self._input(str(Path(cfg["manifest_dir"]) / f"{prefix}_{split}.jsonl.gz"))
            with gzip.open(manifest, "rt", encoding="utf-8") as stream:
                for line in stream:
                    rows += 1
                    size += len(line)
                    if rows > _MAX_REQUESTS or size > _MAX_BYTES or time.monotonic() > deadline:
                        msg = "Speech manifest exceeds supported admission budget"
                        raise ValueError(msg)
                    cut = json.loads(line)
                    admitted = self._admit_cut(cut, samplerate)
                    if admitted.id in seen_cut_ids:
                        msg = (
                            "Duplicate cut ID across native job inputs; cut IDs must be unique across all dset_splits"
                        )
                        raise ValueError(msg)
                    seen_cut_ids.add(admitted.id)
                    _admit_export_speaker(admitted.supervisions[0].speaker, bool(cfg.get("save_spk")))
                    speaker = _eligible_speaker(admitted, cfg)
                    if speaker is not None:
                        speakers[speaker] += 1
        capacity = sum(count >= cfg.get("min_spk_utt", 5) for count in speakers.values())
        if capacity < cfg.get("min_max_spk", [2, 4])[1]:
            msg = "Insufficient eligible aligned speakers for the requested native speaker range"
            raise ValueError(msg)


def write_fastmss_requests(  # noqa: PLR0913 - explicit dependency bindings
    config_paths: list[str],
    manifest_path: str,
    *,
    native_inputs_path: str,
    engine_path: str,
    recipe_path: str,
    worker_lock_path: str,
    overrides: list[str] | None = None,
    request_key: str = "fastmss_request",
) -> int:
    """Write one planning record per meeting; no waveform generation or inspection.

    Explicit YAML lists and inherited local defaults/ordinary Hydra overrides are
    supported. Directory/glob expansion and custom workers remain native-API-only.
    Put this immutable request manifest outside the declared input trees.
    """
    scope = _NativeInputs(
        native_inputs_path=native_inputs_path,
        engine_path=engine_path,
        recipe_path=recipe_path,
        worker_lock_path=worker_lock_path,
    )
    scope._check_roots()
    destination = Path(manifest_path).expanduser().resolve()
    if any(
        destination.is_relative_to(Path(root))
        for root in (scope.native_inputs_path, scope.engine_path, scope.recipe_path)
    ):
        msg = "Request manifest must be outside the immutable native input trees"
        raise ValueError(msg)
    if not config_paths or not request_key:
        msg = "Provide at least one explicit YAML and a nonempty request key"
        raise ValueError(msg)
    dependency_identity = scope._identity()
    plans = [scope._plan(path, list(overrides or [])) for path in config_paths]
    if len({_digest(plan) for plan in plans}) != len(plans):
        msg = "Duplicate native jobs would create duplicate meeting request identities"
        raise ValueError(msg)
    count = sum(plan["native_config"]["n_meetings"] for plan in plans)
    if count > _MAX_REQUESTS:
        msg = "Request manifest exceeds 100,000 meeting requests"
        raise ValueError(msg)
    if scope._identity() != dependency_identity:
        msg = "Native inputs changed during planning"
        raise ValueError(msg)
    scope._write_worker_lock(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        for plan in plans:
            job = {**plan, "dependency_identity": dependency_identity}
            job_id = _digest(job)
            for index in range(plan["native_config"]["n_meetings"]):
                request = {**job, "job_id": job_id, "meeting_index": index, "request_id": f"{job_id}:{index}"}
                stream.write(json.dumps({request_key: request}) + "\n")
    return count


@dataclass(kw_only=True)
class FastMSSRequests(_NativeInputs, AgentReady, ProcessingStage[EmptyTask, AudioTask]):
    """Finite standard manifest source: one selected row means one meeting request.

    Input waveform statistics are uninspected. The native speech/engine/recipe
    dependencies are declared on source and generator, covering checkpoint reuse.
    """

    manifest_path: str
    request_key: str = "fastmss_request"
    name: str = "fastmss_requests"
    KEY_ROLE_OVERRIDES: ClassVar[dict[str, Role]] = {"request_key": "unknown"}
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(lifecycle_side_effects=True, per_row_independent=False)
    )

    def __post_init__(self) -> None:
        self._check_roots()
        self._check_worker()
        path = Path(self.manifest_path).expanduser()
        if not path.is_file() or path.suffix not in {".jsonl", ".json"}:
            msg = "Use one existing local JSONL request manifest"
            raise ValueError(msg)

    def process(self, task: EmptyTask) -> list[AudioTask]:
        rows = ManifestReaderStage().process(
            FileGroupTask(
                dataset_name=task.dataset_name,
                data=[self.manifest_path],
                _metadata=task._metadata.copy(),
                _stage_perf=list(task._stage_perf),
            )
        )
        seen = set()
        for row in rows:
            request = row.data.get(self.request_key)
            if not isinstance(request, dict) or not isinstance(request.get("request_id"), str):
                msg = "Manifest row must contain an identified meeting request object"
                raise TypeError(msg)
            if request["request_id"] in seen or len(seen) >= _MAX_REQUESTS:
                msg = "Duplicate or excessive meeting request records"
                raise ValueError(msg)
            seen.add(request["request_id"])
        return rows

    def num_workers(self) -> int:
        return 1

    def describe(self) -> StageContract:
        return StageContract(
            writes=IOSpec(data_keys=[self.request_key]),
            preserves_upstream_keys=False,
            cardinality="1:N fan-out",
            gates=Gates(lifecycle_side_effects=True, per_row_independent=False),
        )


@dataclass(kw_only=True)
class FastMSSRequestStage(_NativeInputs, AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """Fulfil meeting requests through native bulk generation, then replace metadata.

    ``audio_filepath`` names the complete mixed recording, ``duration`` is its
    elapsed seconds including overlap/silence, ``text`` joins supervision text in
    start/ID order and ``rttm_filepath`` is reference speaker timing. These are
    whole-meeting fields, never isolated speaker tracks or disjoint ASR chunks.
    Native extra annotations survive under ``native_annotations_key`` rather
    than introducing undeclared top-level fields. ``request_id_key`` binds one generated
    meeting to one planning request; it is not a source utterance identity.

    Batch size changes native invocation boundaries and stochastic results; no
    bit-exact equivalence across batches or worker PIDs is claimed. Missing or
    excess meetings fail the batch instead of being counted as filtered requests.
    """

    output_dir: str
    request_key: str = "fastmss_request"
    request_id_key: str = "fastmss_request_id"
    native_annotations_key: str = "fastmss_annotations"
    audio_filepath_key: str = "audio_filepath"
    duration_key: str = "duration"
    text_key: str = "text"
    rttm_filepath_key: str = "rttm_filepath"
    batch_size: int = 32
    jobs_per_config: int = 1
    timeout_seconds: float = 3600.0
    name: str = "fastmss_request"
    is_resumable: bool = False
    KEY_ROLE_OVERRIDES: ClassVar[dict[str, Role]] = {
        "text_key": "reference_text",
        "request_key": "unknown",
        "request_id_key": "unknown",
        "native_annotations_key": "unknown",
        "rttm_filepath_key": "output_path",
    }
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(writes_to_disk=True, output_path_params=["output_dir"], per_row_independent=False),
        error_policy="fail",
    )

    def __post_init__(self) -> None:
        self._check_roots()
        self._check_worker()
        self.output_dir = str(Path(self.output_dir).expanduser().resolve())
        output = Path(self.output_dir)
        if any(
            output.is_relative_to(Path(root)) or Path(root).is_relative_to(output)
            for root in (self.native_inputs_path, self.engine_path, self.recipe_path)
        ):
            msg = "Output must be disjoint from every immutable native input tree"
            raise ValueError(msg)
        keys = [
            self.request_key,
            self.request_id_key,
            self.native_annotations_key,
            self.audio_filepath_key,
            self.duration_key,
            self.text_key,
            self.rttm_filepath_key,
        ]
        if not all(keys) or len(set(keys)) != len(keys):
            msg = "Request and generated output keys must be nonempty and distinct"
            raise ValueError(msg)
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or not 1 <= self.batch_size <= _MAX_BATCH
        ):
            msg = "batch_size must be an integer between 1 and 32"
            raise ValueError(msg)
        if not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= _MAX_TIMEOUT:
            msg = "timeout_seconds must be positive and at most 3600"
            raise ValueError(msg)
        FastMSSSimulationStage(output_dir=self.output_dir, jobs_per_config=self.jobs_per_config)
        self.resources = Resources(cpus=float(self.jobs_per_config))

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def _admit_request(self, request: object, identity: str, plans: dict) -> dict:
        if not isinstance(request, dict):
            msg = "Missing meeting request object"
            raise TypeError(msg)
        if request.get("dependency_identity") != identity:
            msg = "Stale meeting request dependencies; prepare planning metadata again"
            raise ValueError(msg)
        config_path, overrides = request.get("config_path", ""), request.get("overrides", [])
        cache_key = _digest([config_path, overrides])
        if cache_key not in plans:
            plans[cache_key] = self._plan(config_path, overrides)
        plan = plans[cache_key]
        job = {**plan, "dependency_identity": identity}
        job_id = _digest(job)
        index = request.get("meeting_index")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < plan["native_config"]["n_meetings"]
        ):
            msg = "Meeting request index exceeds native capacity"
            raise ValueError(msg)
        expected = {**job, "job_id": job_id, "meeting_index": index, "request_id": f"{job_id}:{index}"}
        if request != expected:
            msg = "Stale meeting request metadata; prepare planning metadata again"
            raise ValueError(msg)
        return expected

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        if len(tasks) > self.batch_size:
            msg = "Batch exceeds the declared native generation bound"
            raise ValueError(msg)
        identity = self._identity()
        groups = defaultdict(list)
        seen = set()
        plans = {}
        for position, task in enumerate(tasks):
            request = self._admit_request(task.data.get(self.request_key), identity, plans)
            if request["request_id"] in seen:
                msg = "Duplicate meeting request identity"
                raise ValueError(msg)
            seen.add(request["request_id"])
            groups[request["job_id"]].append((position, task, request))
        # Admission of every selected row finishes before any generation writes.
        outputs = {}
        for group in groups.values():
            request = group[0][2]
            simulation = FastMSSSimulationStage(
                output_dir=self.output_dir,
                fastmss_root=str(Path(self.recipe_path).parent),
                isolate_work_dirs=True,
                max_meetings=len(group),
                expected_meetings=len(group),
                expected_duration=request["native_config"]["duration"],
                planned_config=request["native_config"],
                jobs_per_config=self.jobs_per_config,
                timeout_seconds=self.timeout_seconds,
            )
            reader = FastMSSManifestReaderStage(
                audio_filepath_key=self.audio_filepath_key,
                duration_key=self.duration_key,
                text_key=self.text_key,
                rttm_filepath_key=self.rttm_filepath_key,
            )
            manifest_tasks = simulation.process(
                FileGroupTask(dataset_name=group[0][1].dataset_name, data=[request["config_path"]])
            )
            rows = [row for manifest in manifest_tasks for row in reader.process(manifest)]
            if len(rows) != len(group):
                msg = f"Unfulfilled meeting requests: requested {len(group)}, generated {len(rows)}; see {self.output_dir}"
                raise RuntimeError(msg)
            # Native worker completion order is arbitrary; preserve native identifiers.
            rows.sort(key=lambda row: row.data[self.audio_filepath_key])
            for (position, parent, req), row in zip(group, rows, strict=True):
                core_keys = [self.audio_filepath_key, self.duration_key, self.text_key, self.rttm_filepath_key]
                values = {key: row.data[key] for key in core_keys}
                annotations = {key: value for key, value in row.data.items() if key not in core_keys}
                if self.request_id_key in annotations or self.native_annotations_key in annotations:
                    msg = "Generated output key conflicts with an existing native annotation"
                    raise ValueError(msg)
                row.data.clear()
                row.data.update(
                    {**values, self.request_id_key: req["request_id"], self.native_annotations_key: annotations}
                )
                row._metadata = parent._metadata.copy()
                row._stage_perf = list(parent._stage_perf)
                row._source_id = parent._source_id
                outputs[position] = row
        if self._identity() != identity:
            msg = "Native dependencies changed during generation; results are not reusable"
            raise RuntimeError(msg)
        return [outputs[index] for index in range(len(tasks))]

    def describe(self) -> StageContract:
        return StageContract(
            reads=IOSpec(data_keys=[self.request_key]),
            writes=IOSpec(
                data_keys=[
                    self.audio_filepath_key,
                    self.duration_key,
                    self.text_key,
                    self.rttm_filepath_key,
                    self.request_id_key,
                    self.native_annotations_key,
                ]
            ),
            preserves_upstream_keys=False,
            preserves_upstream_segment_keys=False,
            cardinality="1:1",
            gates=Gates(writes_to_disk=True, output_path_params=["output_dir"], per_row_independent=False),
            error_policy="fail",
        )
