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

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, StageContract, StaticHints
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class ManifestWriterStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """Append AudioTasks to a driver-prepared JSONL manifest.

    Audio-agent preparation truncates once before workers are serialized. Worker and
    node setup only verify the shared run reservation. Standalone setup() retains
    its historical truncate behavior when no driver reservation was prepared.
    """

    output_path: str
    name: str = "manifest_writer"

    # ``output_path`` is required, so static discovery cannot instantiate this
    # stage. Publish invariant sink behavior so an agent never mistakes it for
    # a pure pass-through before it has configuration values.
    # Agent runners stage terminal output and promote it only after verified success.
    AGENT_ATOMIC_OUTPUT_PARAMS: ClassVar[tuple[str, ...]] = ("output_path",)

    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(
            writes_to_disk=True,
            output_path_params=["output_path"],
            lifecycle_side_effects=True,
            requires_serializable_input=True,
            per_row_independent=True,
        )
    )

    def _resolve_output(self) -> None:
        self._fs, self._path = url_to_fs(self.output_path)
        parent_dir = "/".join(self._path.split("/")[:-1])
        if parent_dir:
            self._fs.makedirs(parent_dir, exist_ok=True)

    def _output_digest(self) -> str:
        digest = hashlib.sha256()
        with self._fs.open(self._path, "rb") as output:
            for chunk in iter(lambda: output.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _checkpoint_digest(self) -> str | None:
        from nemo_curator.utils.resumability_actor import METADATA_DIRNAME

        files = sorted((Path(self._checkpoint_path) / METADATA_DIRNAME).glob("*.mdb"))
        if not files:
            return None
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.name.encode())
            with path.open("rb") as state:
                for chunk in iter(lambda: state.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    def _bind_checkpoint(self) -> str:
        from nemo_curator.utils.resumability_actor import METADATA_DIRNAME

        marker = Path(self._checkpoint_path) / METADATA_DIRNAME / "audio_manifest_pipeline.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        binding = {"pipeline": self._pipeline_identity, "token": uuid.uuid4().hex}
        try:
            with marker.open("x", encoding="utf-8") as output:
                json.dump(binding, output)
                output.flush()
                os.fsync(output.fileno())
        except FileExistsError:
            with marker.open(encoding="utf-8") as source:
                binding = json.load(source)
        if binding.get("pipeline") != self._pipeline_identity or not binding.get("token"):
            msg = "ManifestWriterStage checkpoint belongs to a different pipeline"
            raise RuntimeError(msg)
        return binding["token"]

    def _resume_provenance(self) -> dict[str, str | None]:
        return {
            "checkpoint": self._checkpoint_path,
            "checkpoint_token": self._checkpoint_token,
            "pipeline": self._pipeline_identity,
            "state": self._checkpoint_digest(),
            "output": self._output_digest(),
        }

    def prepare_on_driver(
        self, *, checkpoint_path: str | Path | None = None, pipeline_identity: str | None = None
    ) -> None:
        """Reserve output; append only to a verified successful run of this pipeline."""
        self._resolve_output()
        token = uuid.uuid4().hex
        with self._fs.open(f"{self._path}._RUN", "x", encoding="utf-8") as owner:
            owner.write(token)
        self._run_token = token
        self._checkpoint_path = str(Path(checkpoint_path).resolve()) if checkpoint_path is not None else None
        self._pipeline_identity = pipeline_identity
        try:
            resume = False
            if checkpoint_path is not None:
                state = self._checkpoint_digest()
                exists = self._fs.exists(self._path)
                occupied = exists and self._fs.size(self._path) > 0
                if not pipeline_identity:
                    msg = "ManifestWriterStage needs pipeline_identity with checkpoint_path"
                    raise ValueError(msg)  # noqa: TRY301 - release the owned reservation
                marker = Path(self._checkpoint_path) / ".nemo_curator_metadata" / "audio_manifest_pipeline.json"
                self._checkpoint_token = None
                if marker.exists():
                    with marker.open(encoding="utf-8") as source:
                        self._checkpoint_token = json.load(source).get("token")
                if state is not None or occupied:
                    provenance = None
                    if self._fs.exists(f"{self._path}._RESUME"):
                        with self._fs.open(f"{self._path}._RESUME", "r", encoding="utf-8") as record:
                            provenance = json.load(record)
                    if not exists or not pipeline_identity or state is None or provenance != self._resume_provenance():
                        msg = "ManifestWriterStage cannot authenticate checkpoint, pipeline, and output for resume; use fresh output and checkpoint paths"
                        raise RuntimeError(msg)  # noqa: TRY301 - release the owned reservation on failure
                    resume = True
                self._checkpoint_token = self._bind_checkpoint()
            with self._fs.open(self._path, "a" if resume else "w", encoding="utf-8"):
                pass
            # A prior successful run must not authenticate partially rewritten output.
            if self._fs.exists(f"{self._path}._RESUME"):
                self._fs.rm(f"{self._path}._RESUME")
            self._write_run_owner()
        except Exception:
            self.abort_on_driver()
            raise

    def _output_identity(self) -> dict[str, str]:
        protocols = self._fs.protocol
        protocols = (protocols,) if isinstance(protocols, str) else protocols
        if {"file", "local"}.intersection(protocols):
            stat = os.stat(self._path)
            # Shared mounts can expose different device numbers on different hosts.
            return {"inode": str(stat.st_ino), "ctime_ns": str(stat.st_ctime_ns), "size": str(stat.st_size)}
        self._fs.invalidate_cache(self._path)
        info = self._fs.info(self._path)
        identity = {
            key: str(info[key])
            for key in ("size", "etag", "ETag", "version_id", "VersionId", "generation", "created")
            if key in info and info[key] not in (None, "")
        }
        if not set(identity) - {"size"}:
            msg = "ManifestWriterStage filesystem lacks artifact identity metadata for driver-prepared output"
            raise RuntimeError(msg)
        return identity

    def _read_run_owner(self) -> dict[str, Any]:
        try:
            with self._fs.open(f"{self._path}._RUN", "r", encoding="utf-8") as owner:
                text = owner.read()
        except FileNotFoundError:
            return {}
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            # Initial reservation contains only the token, before output preparation.
            return {"token": text}
        return value if isinstance(value, dict) else {}

    def _write_run_owner(self) -> None:
        payload = {"token": self._run_token, "artifact": self._output_identity()}
        with self._fs.open(f"{self._path}._RUN", "w", encoding="utf-8") as owner:
            json.dump(payload, owner)

    def _verify_run(self) -> None:
        try:
            owner = self._read_run_owner()
            if owner.get("token") == self._run_token and owner.get("artifact") == self._output_identity():
                return
        except FileNotFoundError:
            pass
        msg = f"ManifestWriterStage cannot see its driver-prepared shared output at {self.output_path!r}"
        raise RuntimeError(msg)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        self._resolve_output()
        if getattr(self, "_run_token", None) is not None:
            self._verify_run()
        else:
            if self._fs.exists(f"{self._path}._RUN"):
                msg = "ManifestWriterStage cannot truncate output owned by an active pipeline run"
                raise RuntimeError(msg)
            with self._fs.open(self._path, "w", encoding="utf-8"):
                pass
        logger.info(f"ManifestWriterStage: writing to {self.output_path}")

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        self._resolve_output()
        if getattr(self, "_run_token", None) is not None:
            self._verify_run()

    def finalize(self) -> None:
        if getattr(self, "_run_token", None) is not None:
            self._verify_run()
            if self._checkpoint_path is not None and self._pipeline_identity is not None:
                provenance = self._resume_provenance()
                if provenance["state"] is not None:
                    with self._fs.open(f"{self._path}._RESUME", "w", encoding="utf-8") as record:
                        json.dump(provenance, record, sort_keys=True)
            self.abort_on_driver()

    def reset_for_retry(self) -> None:
        """Release this agent run before the driver prepares its next attempt."""
        self.abort_on_driver()

    def abort_on_driver(self) -> None:
        """Release only this run's ownership; retain partial output on failure."""
        if getattr(self, "_run_token", None) is None:
            return
        self._resolve_output()
        owner_path = f"{self._path}._RUN"
        if self._fs.exists(owner_path):
            if self._read_run_owner().get("token") != self._run_token:
                return
            self._fs.rm(owner_path)
        self._run_token = None

    def write_jsonl_batch(self, records: str) -> None:
        """Append serialized JSONL records, verifying ownership once per batch.

        Callers provide newline-terminated records produced by their serializer.
        Empty batches leave both the output and its ownership marker untouched.
        """
        if not records:
            return
        if getattr(self, "_run_token", None) is not None:
            self._verify_run()
        with self._fs.open(self._path, "a", encoding="utf-8") as output:
            output.write(records)
        if getattr(self, "_run_token", None) is not None:
            self._write_run_owner()

    def process(self, task: AudioTask) -> AudioTask:
        self.write_jsonl_batch(json.dumps(task.data, ensure_ascii=False) + "\n")
        return AudioTask(
            dataset_name=task.dataset_name,
            data=task.data,
            _metadata=task._metadata,
            _stage_perf=list(task._stage_perf),
        )

    def num_workers(self) -> int | None:
        return 1

    def describe(self) -> StageContract:
        return StageContract(
            gates=Gates(
                writes_to_disk=True,
                output_path_params=["output_path"],
                lifecycle_side_effects=True,
                # Serializes task.data as-is via json.dumps; a resident tensor
                # (e.g. a waveform) will crash it. Stop carrying the tensor
                # before this AudioTask sink, or convert to a DocumentBatch and
                # use DocumentBatchJsonlWriterStage instead.
                requires_serializable_input=True,
                # Appends each row as it arrives; a row's line is its own contents.
                per_row_independent=True,
            ),
        )
