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

"""Local, finite native-job requests using the agent's existing JSONL protocol.

One request is one complete YAML invocation, not one speaker or meeting. Native
n_meetings, duration, seeds and generation settings are never rewritten. Sampling
N requests bounds invocations; their declared n_meetings bounds the output fanout.
The source has no audio-quality profile: its rows are planning metadata.

Prepare requests outside the pipeline with write_fastmss_requests, then build
FastMSSRequests(manifest_path=..., **roots) followed by
FastMSSRequestStage(output_dir=..., **roots). The explicit roots are
native_inputs_path (all local input data/configs), engine_path (fastmss/), and
recipe_path (recipes/). The same roots are declared on both stages, so source
checkpoints cannot hide a changed native dependency. Keep them read-only during
execution. Put generated output outside every input tree.

Agent compatibility currently covers standard recipes with local file-backed
MonoCuts, noise and fitting inputs in native_inputs_path. The engine and recipe
trees are explicit dependencies too. Custom worker environments, executable audio
sources, dynamic resolvers (except FASTMSS_ROOT), and config search paths outside these trees remain unsupported here;
the original FastMSS YAML API remains available independently.
"""

import gzip
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, Role, StageContract, StaticHints
from nemo_curator.stages.audio.common import ManifestReader, ManifestReaderStage
from nemo_curator.stages.audio.simulation.fastmss import FastMSSManifestReaderStage, FastMSSSimulationStage
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask, FileGroupTask


@dataclass(kw_only=True)
class _NativeInputs:
    native_inputs_path: str
    engine_path: str
    recipe_path: str

    def _check_native_roots(self) -> None:
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
            msg = "Agent requests require the standard sibling fastmss/ and recipes/sim.py installation"
            raise ValueError(msg)

    def _input(self, value: str, *, recipe: bool = False) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(self.recipe_path).parent / path  # Native worker cwd, not YAML-relative.
        path = path.resolve()
        roots = [Path(self.native_inputs_path), *([Path(self.recipe_path)] if recipe else [])]
        if not path.exists() or not any(path.is_relative_to(root) for root in roots):
            msg = f"Native dependency is missing or outside its declared input tree: {path}"
            raise ValueError(msg)
        return path

    def spec(self, config_path: str) -> dict:
        """Resolve native defaults read-only and prove that all actual reads are tracked."""
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        config = self._input(config_path)
        if config.suffix != ".yaml":
            msg = "Agent request configs require .yaml, matching the native Hydra interface"
            raise ValueError(msg)
        previous = os.environ.get("FASTMSS_ROOT")
        os.environ["FASTMSS_ROOT"] = str(Path(self.recipe_path).parent)
        try:
            with initialize_config_dir(config_dir=str(config.parent), version_base=None):
                composed = compose(config_name=config.stem, return_hydra_config=True)
                for source in composed.hydra.runtime.config_sources:
                    if source.schema == "file":
                        self._input(source.path, recipe=True)
                    elif source.schema == "pkg" and source.provider != "hydra":
                        msg = "External package defaults are unsupported; use declared local recipe trees"
                        raise ValueError(msg)
                OmegaConf.set_struct(composed, False)
                del composed["hydra"]
                raw = json.dumps(OmegaConf.to_container(composed, resolve=False))
                if re.search(r"\$\{[^{}]*:", raw.replace("${oc.env:FASTMSS_ROOT}", "")):
                    msg = "Dynamic resolvers are unsupported by request identity except oc.env:FASTMSS_ROOT"
                    raise ValueError(msg)
                cfg = OmegaConf.to_container(composed, resolve=True)
        finally:
            if previous is None:
                os.environ.pop("FASTMSS_ROOT", None)
            else:
                os.environ["FASTMSS_ROOT"] = previous
        count, duration = cfg.get("n_meetings"), cfg.get("duration")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            msg = "A finite native job requires positive integer n_meetings"
            raise ValueError(msg)
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            msg = "A finite native job requires positive finite duration"
            raise ValueError(msg)
        self._check_native_data(cfg)
        return {"config_path": str(config), "n_meetings": count, "duration": duration, "native_config": cfg}

    def _check_native_data(self, cfg: dict) -> None:  # noqa: C901 - independent native source kinds
        prefix = cfg["manifest_prefix"]
        if not isinstance(prefix, str) or not prefix or "/" in prefix or "\\" in prefix or prefix in {".", ".."}:
            msg = "Isolated requests require a plain source manifest prefix, without path components"
            raise ValueError(msg)
        for split in cfg["dset_splits"]:
            manifest = self._input(str(Path(cfg["manifest_dir"]) / f"{cfg['manifest_prefix']}_{split}.jsonl.gz"))
            with gzip.open(manifest, "rt", encoding="utf-8") as stream:
                for line in stream:
                    cut = json.loads(line)
                    if cut.get("type") != "MonoCut" or "recording" not in cut:
                        msg = "Agent requests require embedded file-backed MonoCut recordings"
                        raise ValueError(msg)
                    for source in cut["recording"]["sources"]:
                        if source["type"] != "file":
                            msg = "Executable or remote recording sources are unsupported by agent requests"
                            raise ValueError(msg)
                        self._input(source["source"])
        dependencies = []
        if cfg.get("add_noise"):
            dependencies.extend(cfg.get("noise_folders") or [])
        if cfg.get("hmm_fit_transitions_to"):
            dependencies.append(cfg["hmm_fit_transitions_to"])
        for value in dependencies:
            dependency = self._input(value)
            if dependency.is_dir():
                for path in dependency.rglob("*"):
                    if path.is_file():
                        self._input(str(path))


def write_fastmss_requests(  # noqa: PLR0913 - explicit dependency roots
    config_paths: list[str],
    manifest_path: str,
    *,
    native_inputs_path: str,
    engine_path: str,
    recipe_path: str,
    request_key: str = "fastmss_request",
) -> int:
    """Prepare finite planning metadata; generation still happens inside the pipeline.

    All native input files must belong to the declared local tree. The existing
    agent dependency protocol fingerprints that tree and both toolkit trees;
    changing speech, YAML/defaults, noise or engine files changes the step key.
    """
    scope = _NativeInputs(native_inputs_path=native_inputs_path, engine_path=engine_path, recipe_path=recipe_path)
    scope._check_native_roots()
    rows = [{request_key: scope.spec(path)} for path in config_paths]
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return len(rows)


@dataclass
class FastMSSRequestReaderStage(ManifestReaderStage):
    """Materialize request metadata as typed tasks; these are not audio recordings."""

    request_key: str = "fastmss_request"
    name: str = "fastmss_request_reader"
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(lifecycle_side_effects=True, per_row_independent=False)
    )
    KEY_ROLE_OVERRIDES: ClassVar[dict[str, Role]] = {"include_files_key": "audio_filepath", "request_key": "unknown"}

    def describe(self) -> StageContract:
        return StageContract(
            writes=IOSpec(data_keys=[self.request_key]),
            preserves_upstream_keys=False,
            cardinality="1:N fan-out",
            gates=Gates(lifecycle_side_effects=True, per_row_independent=False),
        )


@dataclass(kw_only=True)
class FastMSSRequests(_NativeInputs, ManifestReader):
    """Existing manifest protocol over finite native jobs, with tracked native inputs.

    Use as the source, followed by FastMSSRequestStage. Smoke samples complete
    requests, preserving each native job's meeting count. Audio-folder selection
    and changed-file delta sampling are unsupported for request metadata.
    """

    request_key: str = "fastmss_request"
    name: str = "fastmss_requests"
    KEY_ROLE_OVERRIDES: ClassVar[dict[str, Role]] = {"include_files_key": "audio_filepath", "request_key": "unknown"}
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(gates=Gates(per_row_independent=False))

    def __post_init__(self) -> None:
        self._check_native_roots()
        if self.include_files is not None:
            msg = "Select finite request rows, not audio-file include_files"
            raise ValueError(msg)
        super().__post_init__()

    def decompose(self) -> list[ProcessingStage]:
        return [
            FilePartitioningStage(
                file_paths=self.manifest_path, files_per_partition=1, file_extensions=[".jsonl", ".json"]
            ),
            FastMSSRequestReaderStage(request_key=self.request_key),
        ]

    def describe(self) -> StageContract:
        return StageContract(cardinality="1:N fan-out", wrappable=False, gates=Gates(per_row_independent=False))


@dataclass(kw_only=True)
class FastMSSRequestStage(_NativeInputs, AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """Generate one complete native job per request, then emit meeting-level tasks.

    This adapter always isolates caches/writes. It never changes n_meetings,
    duration, seeds, source speaker selection or native generation settings.
    Downstream filters act on generated meetings, not on request counts.
    """

    output_dir: str
    request_key: str = "fastmss_request"
    audio_filepath_key: str = "audio_filepath"
    duration_key: str = "duration"
    text_key: str = "text"
    rttm_filepath_key: str = "rttm_filepath"
    jobs_per_config: int = 1
    timeout_seconds: float = 3600.0
    name: str = "fastmss_request"
    is_resumable: bool = False
    KEY_ROLE_OVERRIDES: ClassVar[dict[str, Role]] = {"request_key": "unknown", "rttm_filepath_key": "output_path"}
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(writes_to_disk=True, output_path_params=["output_dir"], per_row_independent=False)
    )

    def __post_init__(self) -> None:
        self._check_native_roots()
        self.output_dir = str(Path(self.output_dir).expanduser().resolve())
        if any(
            Path(self.output_dir).is_relative_to(Path(getattr(self, name)))
            for name in ("native_inputs_path", "engine_path", "recipe_path")
        ):
            msg = "Generated output must be outside all declared native input trees"
            raise ValueError(msg)
        self.resources = Resources(cpus=float(self.jobs_per_config))

    def process(self, task: AudioTask) -> list[AudioTask]:
        request = task.data[self.request_key]
        if not isinstance(request, dict) or self.spec(request["config_path"]) != request:
            msg = "Request planning metadata differs from the effective native configuration; prepare it again"
            raise ValueError(msg)
        simulation = FastMSSSimulationStage(
            output_dir=self.output_dir,
            fastmss_root=str(Path(self.recipe_path).parent),
            script_path=f"{Path(self.recipe_path).name}/sim.py",
            isolate_work_dirs=True,
            jobs_per_config=self.jobs_per_config,
            timeout_seconds=self.timeout_seconds,
            expected_meetings=request["n_meetings"],
            expected_duration=request["duration"],
        )
        group = FileGroupTask(
            dataset_name=task.dataset_name,
            data=[request["config_path"]],
            _metadata=task._metadata.copy(),
            _stage_perf=list(task._stage_perf),
        )
        reader = FastMSSManifestReaderStage(
            audio_filepath_key=self.audio_filepath_key,
            duration_key=self.duration_key,
            text_key=self.text_key,
            rttm_filepath_key=self.rttm_filepath_key,
        )
        rows = [row for manifest in simulation.process(group) for row in reader.process(manifest)]
        if len(rows) > request["n_meetings"]:
            msg = "Native generation exceeded the request's declared meeting bound"
            raise ValueError(msg)
        return rows

    def describe(self) -> StageContract:
        return StageContract(
            reads=IOSpec(data_keys=[self.request_key]),
            writes=IOSpec(
                data_keys=[self.audio_filepath_key, self.duration_key, self.text_key, self.rttm_filepath_key]
            ),
            preserves_upstream_keys=False,
            cardinality="1:N fan-out",
            gates=Gates(writes_to_disk=True, output_path_params=["output_dir"], per_row_independent=False),
            error_policy="fail",
        )
