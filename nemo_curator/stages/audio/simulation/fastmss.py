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

"""Run the bundled original FastMSS and ingest recording-level manifests."""

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from loguru import logger

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, StageContract, StaticHints
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

    The original Apache-2.0 FastMSS implementation is bundled with Curator.
    Install its optional simulation dependencies in ``python_executable`` as
    described in the tutorial. Source speech and noise datasets are provided by users.

    Args:
        fastmss_root: Optional checkout override; defaults to Curator's bundled upstream snapshot.
        output_dir: Shared local filesystem directory for generated artifacts.
        script_path: Python entry point relative to the selected FastMSS source root.
        python_executable: Worker Python with FastMSS and its dependencies installed.
        jobs_per_config: FastMSS child workers and reserved Curator CPUs per config.
        overrides: Additional Hydra overrides, for example ``duration=60``.
            Output paths, worker count and recording-manifest generation are managed
            by this stage and cannot be overridden here.
        timeout_seconds: Maximum wall time for one config, including preprocessing.
    """

    output_dir: str
    fastmss_root: str | None = None
    script_path: str = "recipes/sim.py"
    python_executable: str = field(default_factory=lambda: sys.executable)
    jobs_per_config: int = 1
    overrides: list[str] = field(default_factory=list)
    timeout_seconds: float = 3600.0
    name: str = "fastmss_simulation"
    is_resumable: bool = False
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(writes_to_disk=True, output_path_params=["output_dir"], per_row_independent=False),
        error_policy="fail",
    )

    def __post_init__(self) -> None:
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
        root = (
            Path(self.fastmss_root)
            if self.fastmss_root is not None
            else Path(__file__).parents[3] / "_vendor" / "fastmss"
        )
        self.fastmss_root = str(root.expanduser().resolve())
        self.output_dir = str(Path(self.output_dir).expanduser().resolve())
        root = Path(self.fastmss_root)
        if not (root / self.script_path).resolve().is_relative_to(root):
            msg = "script_path must resolve inside fastmss_root"
            raise ValueError(msg)
        self.resources = Resources(cpus=float(self.jobs_per_config))

    def describe(self) -> StageContract:
        return StageContract(
            cardinality="1:N fan-out",
            wrappable=False,
            gates=Gates(writes_to_disk=True, output_path_params=["output_dir"], per_row_independent=False),
            error_policy="fail",
        )

    def _run(self, config: Path, run_dir: Path) -> None:
        script = Path(self.fastmss_root) / self.script_path
        command = [
            self.python_executable,
            str(script),
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
        env["FASTMSS_ROOT"] = self.fastmss_root
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (self.fastmss_root, env.get("PYTHONPATH"))))
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
        script = Path(self.fastmss_root) / self.script_path
        if not script.is_file():
            msg = f"FastMSS entry point does not exist: {script}"
            raise FileNotFoundError(msg)
        configs = [Path(path).expanduser().resolve() for path in task.data]
        for config in configs:
            if config.suffix not in {".yaml", ".yml"} or not config.is_file():
                msg = f"Expected an existing local FastMSS YAML config: {config}"
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
class FastMSS(AgentReady, CompositeStage[EmptyTask, AudioTask]):
    """Generate simulated meetings and read one AudioTask per recording.

    Each YAML is one scheduling unit, containing its own ``n_meetings``.
    Multiple YAMLs can run in parallel; one config uses ``jobs_per_config`` CPUs.
    Generated paths and durations refer to entire mixtures, including overlaps,
    not isolated speaker turns. ``text`` is FastMSS's meeting-level reference,
    and ``rttm_filepath`` points to generated diarization reference annotations.
    Additional manifest fields are preserved verbatim by ManifestReaderStage.

    All paths, input speech and the Python environment must be visible
    on every executor node. Failed runs retain artifacts and logs for diagnosis;
    reruns create fresh output directories rather than resuming partial meetings.

    Args:
        config_paths: Local YAML files, directories or globs for independent shards.
        fastmss_root: Optional checkout override; defaults to the bundled original FastMSS.
        output_dir: Shared output directory for mixtures, RTTM and manifests.
        script_path: Hydra Python entry point relative to the checkout.
        python_executable: Python with FastMSS dependencies installed on workers.
        jobs_per_config: CPU workers per FastMSS invocation.
        overrides: Extra Hydra configuration overrides.
        timeout_seconds: Maximum seconds per shard.
    """

    config_paths: str | list[str]
    output_dir: str
    fastmss_root: str | None = None
    script_path: str = "recipes/sim.py"
    python_executable: str = field(default_factory=lambda: sys.executable)
    jobs_per_config: int = 1
    overrides: list[str] = field(default_factory=list)
    timeout_seconds: float = 3600.0
    name: str = "fastmss"
    is_resumable: bool = False
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(writes_to_disk=True, output_path_params=["output_dir"], per_row_independent=False)
    )

    def __post_init__(self) -> None:
        super().__init__()

    def decompose(self) -> list[ProcessingStage]:
        return [
            FilePartitioningStage(
                file_paths=self.config_paths,
                files_per_partition=1,
                file_extensions=[".yaml", ".yml"],
            ),
            FastMSSSimulationStage(
                fastmss_root=self.fastmss_root,
                output_dir=self.output_dir,
                script_path=self.script_path,
                python_executable=self.python_executable,
                jobs_per_config=self.jobs_per_config,
                overrides=self.overrides,
                timeout_seconds=self.timeout_seconds,
            ),
            ManifestReaderStage(),
        ]

    def describe(self) -> StageContract:
        return StageContract(
            cardinality="1:N fan-out",
            wrappable=False,
            gates=Gates(writes_to_disk=True, output_path_params=["output_dir"], per_row_independent=False),
        )
