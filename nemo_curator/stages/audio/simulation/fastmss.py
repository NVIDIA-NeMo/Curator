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

"""Run the installed FastMSS toolkit and ingest recording-level manifests."""

import contextlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from loguru import logger

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, Role, StageContract, StaticHints
from nemo_curator.stages.audio.common import ManifestReaderStage
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask, EmptyTask, FileGroupTask

_MANAGED_OVERRIDES = frozenset(
    {"output_dir", "n_jobs", "stage", "save_rttm", "save_nemo_manifest", "hydra.run.dir", "hydra.job.chdir"}
)


@dataclass
class FastMSSSimulationStage(AgentReady, ProcessingStage[FileGroupTask, FileGroupTask]):
    """Run one FastMSS simulation per input YAML and emit its manifest path.

    Input ``FileGroupTask.data`` contains local YAML paths, not speech records.
    Every YAML defines an independent shard of simulated meetings. Its FastMSS
    source manifests must contain prepared, aligned single-speaker speech.

    Each invocation writes to a fresh subdirectory of ``output_dir``. The
    returned file group names the recording-level ``nemo_manifest.json``;
    RTTM, waveforms and optional per-speaker references stay in that run's
    directory. It does not emit one task per utterance or flatten overlaps.

    Install Curator's ``audio_simulation`` extra in ``python_executable``.
    Source speech and noise datasets are provided by users.

    Args:
        fastmss_root: Optional checkout override; defaults to the installed FastMSS distribution.
        output_dir: Shared local filesystem directory for generated artifacts.
        script_path: Python entry point relative to the selected FastMSS source root.
        python_executable: Worker Python with FastMSS and its dependencies installed.
        jobs_per_config: FastMSS child workers and reserved Curator CPUs per config.
        overrides: Additional Hydra overrides, for example ``duration=60``.
            Output paths, worker count and recording-manifest generation are managed
            by this stage and cannot be overridden here.
        timeout_seconds: Maximum wall time for one config, including preprocessing.
        isolate_work_dirs: Keep YAML-defined preprocessing and RIR caches run-local.
            Required for agent wrapping; false preserves explicit pipeline behavior.
        max_meetings: Positive meeting cap for exactly one .yaml config; -1 is unlimited.
    """

    output_dir: str
    fastmss_root: str | None = None
    script_path: str = "recipes/sim.py"
    python_executable: str = field(default_factory=lambda: sys.executable)
    jobs_per_config: int = 1
    overrides: list[str] = field(default_factory=list)
    timeout_seconds: float = 3600.0
    isolate_work_dirs: bool = False
    max_meetings: int = -1
    expected_meetings: int | None = field(default=None, kw_only=True)
    expected_duration: float | None = field(default=None, kw_only=True)
    name: str = "fastmss_simulation"
    is_resumable: bool = False
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(writes_to_disk=True, output_path_params=["output_dir"], per_row_independent=False),
        error_policy="fail",
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_meetings, bool)
            or not isinstance(self.max_meetings, int)
            or self.max_meetings == 0
            or self.max_meetings < -1
        ):
            msg = "max_meetings must be positive or -1 (unlimited)"
            raise ValueError(msg)
        if self.jobs_per_config < 1:
            msg = "jobs_per_config must be at least 1"
            raise ValueError(msg)
        if self.timeout_seconds <= 0:
            msg = "timeout_seconds must be positive"
            raise ValueError(msg)
        for override in self.overrides:
            key, separator, _ = override.partition("=")
            if not separator or key.startswith(("-", "~")) or key.lstrip("+") in _MANAGED_OVERRIDES:
                msg = f"Unsupported or stage-managed FastMSS override: {override!r}"
                raise ValueError(msg)
        self.output_dir = str(Path(self.output_dir).expanduser().resolve())
        root = Path(self.fastmss_root or ".").expanduser().resolve()
        if Path(self.script_path).is_absolute() or not (root / self.script_path).resolve().is_relative_to(root):
            msg = "script_path must resolve inside fastmss_root"
            raise ValueError(msg)
        if self.fastmss_root is not None:
            self.fastmss_root = str(root)
        self.resources = Resources(cpus=float(self.jobs_per_config))

    def describe(self) -> StageContract:
        return StageContract(
            cardinality="1:N fan-out",
            wrappable=self.isolate_work_dirs,
            gates=Gates(writes_to_disk=True, output_path_params=["output_dir"], per_row_independent=False),
            error_policy="fail",
        )

    def _run(self, config: Path, run_dir: Path) -> None:
        script = str(Path(self.fastmss_root) / self.script_path) if self.fastmss_root else self.script_path
        command = [
            self.python_executable,
            str(Path(__file__).with_name("_runner.py")),
            script,
            *(["--isolate-work-dirs"] if self.isolate_work_dirs else []),
            f"--max-meetings={self.max_meetings}",
            *([f"--expected-meetings={self.expected_meetings}"] if self.expected_meetings is not None else []),
            *([f"--expected-duration={self.expected_duration}"] if self.expected_duration is not None else []),
            *(["--installed-fastmss"] if self.fastmss_root is None else []),
            f"--config-path={config.parent}",
            f"--config-name={config.stem}",
            *self.overrides,
            f"output_dir={json.dumps(str(run_dir))}",
            f"n_jobs={self.jobs_per_config}",
            "stage=0",
            "++save_rttm=true",
            "++save_nemo_manifest=true",
            f"hydra.run.dir={json.dumps(str(run_dir))}",
            "hydra.job.chdir=false",
        ]
        env = os.environ.copy()
        if self.fastmss_root is not None:
            env["FASTMSS_ROOT"] = self.fastmss_root
            env["PYTHONPATH"] = os.pathsep.join(filter(None, (self.fastmss_root, env.get("PYTHONPATH"))))
        if self.isolate_work_dirs:
            env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        log_path = run_dir / "fastmss.log"
        logger.info(f"Simulating FastMSS config {config}; output and log: {run_dir}")
        with (
            log_path.open("w", encoding="utf-8") as log,
            subprocess.Popen(  # noqa: S603 - trusted, explicit Python entry point; no shell
                command,
                cwd=self.fastmss_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            ) as process,
        ):
            try:
                returncode = process.wait(timeout=self.timeout_seconds)
            except BaseException:
                # FastMSS may own multiprocessing children; kill the process group too.
                with contextlib.suppress(ProcessLookupError):
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                process.wait()
                raise
        if returncode:
            msg = f"FastMSS failed with exit code {returncode}; see {log_path}"
            raise RuntimeError(msg)

    def process(self, task: FileGroupTask) -> list[FileGroupTask]:
        script = Path(self.fastmss_root) / self.script_path if self.fastmss_root else None
        if script is not None and not script.is_file():
            msg = f"FastMSS entry point does not exist: {script}"
            raise FileNotFoundError(msg)
        configs = [Path(path).expanduser().resolve() for path in task.data]
        for config in configs:
            if config.suffix != ".yaml" or not config.is_file():
                msg = f"Expected an existing local FastMSS YAML config with .yaml extension: {config}"
                raise ValueError(msg)
        if self.max_meetings > 0 and len(configs) != 1:
            msg = "A bounded generation requires exactly one .yaml config"
            raise ValueError(msg)
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for config in configs:
            run_dir = Path(tempfile.mkdtemp(prefix="fastmss-", dir=output_dir))
            self._run(config, run_dir)
            manifest = run_dir / "nemo_manifest.json"
            if not manifest.is_file() or not manifest.stat().st_size:
                msg = (
                    f"FastMSS did not produce a nonempty recording manifest: {manifest}; see {run_dir / 'fastmss.log'}"
                )
                raise RuntimeError(msg)
            results.append(
                FileGroupTask(
                    dataset_name=task.dataset_name,
                    data=[str(manifest)],
                    _metadata=task._metadata.copy(),
                    _stage_perf=list(task._stage_perf),
                )
            )
        return results


@dataclass
class FastMSSManifestReaderStage(ManifestReaderStage):
    """Read generated whole-meeting references, validating every promised field.

    Text retains all overlapping turns in start/ID order; it is not a sequence
    of non-overlapping ASR chunks. Additional manifest annotations are preserved.
    The four output keys may be renamed without changing their meaning.
    """

    audio_filepath_key: str = "audio_filepath"
    duration_key: str = "duration"
    text_key: str = "text"
    rttm_filepath_key: str = "rttm_filepath"
    name: str = "fastmss_manifest_reader"
    KEY_ROLE_OVERRIDES: ClassVar[Mapping[str, Role]] = {
        "include_files_key": "audio_filepath",
        "rttm_filepath_key": "output_path",
    }

    def process(self, task: FileGroupTask) -> list[AudioTask]:
        keys = {
            "audio_filepath": self.audio_filepath_key,
            "duration": self.duration_key,
            "text": self.text_key,
            "rttm_filepath": self.rttm_filepath_key,
        }
        if len(set(keys.values())) != len(keys):
            msg = "FastMSS output keys must be distinct"
            raise ValueError(msg)
        rows = super().process(task)
        for row in rows:
            data = row.data
            for key in ("audio_filepath", "text", "rttm_filepath"):
                if not isinstance(data.get(key), str) or not data[key].strip():
                    msg = f"FastMSS meeting lacks nonempty {key}"
                    raise ValueError(msg)
            for key in ("audio_filepath", "rttm_filepath"):
                if not Path(data[key]).is_file():
                    msg = f"FastMSS meeting artifact does not exist: {data[key]}"
                    raise ValueError(msg)
            duration = data.get("duration")
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration <= 0
            ):
                msg = "FastMSS meeting requires a finite positive duration in seconds"
                raise ValueError(msg)
            values = {target: data[source] for source, target in keys.items()}
            if any(
                target in data and target not in keys and data[target] != value for target, value in values.items()
            ):
                msg = "FastMSS output key conflicts with an existing annotation"
                raise ValueError(msg)
            # AudioTask.data must retain its attribute-access mapping for downstream validation.
            for key in keys:
                data.pop(key, None)
            data.update(values)
        return rows

    def describe(self) -> StageContract:
        return StageContract(
            writes=IOSpec(
                data_keys=[self.audio_filepath_key, self.duration_key, self.text_key, self.rttm_filepath_key]
            ),
            cardinality="1:N fan-out",
            gates=Gates(lifecycle_side_effects=True, per_row_independent=True),
        )


@dataclass
class FastMSS(AgentReady, CompositeStage[EmptyTask, AudioTask]):
    """Generate simulated meetings and read one AudioTask per recording.

    Each YAML is one scheduling unit, containing its own ``n_meetings``.
    Multiple YAMLs can run in parallel; one config uses ``jobs_per_config`` CPUs.
    Generated paths and durations refer to entire mixtures, including overlaps,
    not isolated speaker turns. ``text`` joins generated supervision transcripts
    in start-time order (supervision ID breaks ties), including overlapping turns,
    and ``rttm_filepath`` points to generated diarization reference annotations.
    Additional manifest fields are preserved verbatim by ManifestReaderStage.

    All paths, input speech and the Python environment must be visible
    on every executor node. Failed runs retain artifacts and logs for diagnosis;
    reruns create fresh output directories rather than resuming partial meetings.

    Args:
        config_paths: Local YAML files, directories or globs for independent shards.
        fastmss_root: Optional checkout override; defaults to the installed FastMSS distribution.
        output_dir: Shared output directory for mixtures, RTTM and manifests.
        script_path: Hydra Python entry point relative to the checkout.
        python_executable: Python with FastMSS dependencies installed on workers.
        jobs_per_config: CPU workers per FastMSS invocation.
        overrides: Extra Hydra configuration overrides.
        timeout_seconds: Maximum seconds per shard.
        isolate_work_dirs: Redirect all FastMSS caches under output_dir for agent execution.
        max_meetings: Positive total cap requiring one local .yaml file; -1 uses YAML counts.
        audio_filepath_key: Output key for the meeting waveform path.
        duration_key: Output key for whole-meeting duration in seconds.
        text_key: Output key for joined reference transcripts including overlaps.
        rttm_filepath_key: Output key for the diarization reference path.
    """

    config_paths: str | list[str]
    output_dir: str
    fastmss_root: str | None = None
    script_path: str = "recipes/sim.py"
    python_executable: str = field(default_factory=lambda: sys.executable)
    jobs_per_config: int = 1
    overrides: list[str] = field(default_factory=list)
    timeout_seconds: float = 3600.0
    isolate_work_dirs: bool = False
    max_meetings: int = -1
    audio_filepath_key: str = "audio_filepath"
    duration_key: str = "duration"
    text_key: str = "text"
    rttm_filepath_key: str = "rttm_filepath"
    name: str = "fastmss"
    KEY_ROLE_OVERRIDES: ClassVar[Mapping[str, Role]] = {"rttm_filepath_key": "output_path"}
    is_resumable: bool = False
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(writes_to_disk=True, output_path_params=["output_dir"], per_row_independent=False)
    )

    def __post_init__(self) -> None:
        if self.max_meetings > 0 and (not isinstance(self.config_paths, str) or not Path(self.config_paths).is_file()):
            msg = "A bounded generation requires exactly one local .yaml file"
            raise ValueError(msg)
        FastMSSSimulationStage(
            output_dir=self.output_dir,
            max_meetings=self.max_meetings,
            jobs_per_config=self.jobs_per_config,
            overrides=self.overrides,
            timeout_seconds=self.timeout_seconds,
            script_path=self.script_path,
            fastmss_root=self.fastmss_root,
        )
        paths = [self.config_paths] if isinstance(self.config_paths, str) else self.config_paths
        if any(
            Path(path).suffix == ".yml" or (Path(path).is_file() and Path(path).suffix != ".yaml") for path in paths
        ):
            msg = "FastMSS supports .yaml configs; .yml is unsupported by Hydra"
            raise ValueError(msg)
        super().__init__()

    def decompose(self) -> list[ProcessingStage]:
        return [
            FilePartitioningStage(
                file_paths=self.config_paths,
                files_per_partition=1,
                file_extensions=[".yaml"],
            ),
            FastMSSSimulationStage(
                fastmss_root=self.fastmss_root,
                output_dir=self.output_dir,
                script_path=self.script_path,
                python_executable=self.python_executable,
                jobs_per_config=self.jobs_per_config,
                overrides=self.overrides,
                timeout_seconds=self.timeout_seconds,
                isolate_work_dirs=self.isolate_work_dirs,
                max_meetings=self.max_meetings,
            ),
            FastMSSManifestReaderStage(
                audio_filepath_key=self.audio_filepath_key,
                duration_key=self.duration_key,
                text_key=self.text_key,
                rttm_filepath_key=self.rttm_filepath_key,
            ),
        ]

    def describe(self) -> StageContract:
        return StageContract(
            cardinality="1:N fan-out",
            wrappable=False,
            gates=Gates(writes_to_disk=True, output_path_params=["output_dir"], per_row_independent=False),
        )
