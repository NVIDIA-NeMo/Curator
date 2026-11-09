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
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Any, ClassVar, Literal
from urllib.parse import urlsplit

from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, StageContract, StaticHints
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class ManifestCheckpointStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """Persist a reusable, metadata-only AudioTask boundary as JSONL.

    Pipeline.run(checkpoint_path=...) tracks completed source partitions for
    rerunning the same pipeline. This stage saves intermediate rows so a later
    pipeline can reuse computed scores when downstream filtering changes.

    This stage is an intermediate checkpoint, not a terminal user deliverable. It
    serializes complete ``task.data`` rows and passes each task downstream with
    its dataset name, metadata, and performance records preserved. Waveform
    tensors and other non-JSON values are illegal at this boundary.

    The checkpoint is local-only and single-worker. ``setup()`` exclusively
    reserves a new destination and refuses to overwrite any existing file.

    Args:
        output_path: Required local destination for checkpoint JSONL.
        retention_sec: Advisory retention in seconds. Defaults to 0, meaning
            user-managed with no automatic expiry; must be non-negative.
        owner: Ownership recorded with the checkpoint policy. ``"user"`` is the
            conservative default; ``"project"`` means the project operator owns
            retention. Neither value enables automatic deletion.
        planning_provenance: Internal marker for a reusable-pipeline candidate.
            Such recipes require exact-hash approval and authoritative smoke.
    """

    REUSABLE_PIPELINE_PROVENANCE: ClassVar[str] = "reusable_pipeline_v1"

    output_path: str
    retention_sec: int = 0
    owner: Literal["user", "project"] = "user"
    planning_provenance: Literal["reusable_pipeline_v1"] | None = None
    name: str = "manifest_checkpoint"

    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(
            writes_to_disk=True,
            output_path_params=["output_path"],
            lifecycle_side_effects=True,
            requires_serializable_input=True,
            per_row_independent=True,
        ),
        description="Persist a complete metadata checkpoint without ending the pipeline",
    )

    def __post_init__(self) -> None:
        if urlsplit(self.output_path).scheme:
            msg = "ManifestCheckpointStage output_path must be a plain local path, not a URI"
            raise ValueError(msg)
        if self.retention_sec < 0:
            msg = "retention_sec must be non-negative"
            raise ValueError(msg)
        if self.planning_provenance not in {None, self.REUSABLE_PIPELINE_PROVENANCE}:
            msg = f"planning_provenance must be None or {self.REUSABLE_PIPELINE_PROVENANCE!r}"
            raise ValueError(msg)
        self._reservation_owned = False
        self._reservation_identity: tuple[int, int, int] | None = None
        self._reservation_token = uuid.uuid4().hex
        self._checkpoint_rows_written = 0
        self._checkpoint_bytes_written = 0

    def _resolve_output(self) -> None:
        self._fs, self._path = url_to_fs(self.output_path)
        parent_dir = "/".join(self._path.split("/")[:-1])
        if parent_dir:
            self._fs.makedirs(parent_dir, exist_ok=True)

    def prepare_on_driver(self, **_run_context: object) -> None:
        self._reserve_checkpoint()
        self._driver_prepared = True

    def _verify_reservation(self) -> None:
        self._resolve_output()
        owner = self._read_retry_owner()
        if owner is not None and owner.get("token") == self._reservation_token and self._fs.exists(self._path):
            stat = os.stat(self._path)
            # Device numbers can differ across hosts mounting the same shared filesystem.
            if (stat.st_ino, stat.st_ctime_ns, stat.st_size) == (
                owner.get("st_ino"),
                owner.get("st_ctime_ns"),
                owner.get("st_size"),
            ):
                self._reservation_owned = True
                self._reservation_identity = (stat.st_dev, stat.st_ino, stat.st_ctime_ns)
                return
        msg = f"ManifestCheckpointStage cannot verify its shared reservation at {self.output_path!r}"
        raise RuntimeError(msg)

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        if getattr(self, "_driver_prepared", False):
            self._verify_reservation()

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if getattr(self, "_driver_prepared", False):
            self._verify_reservation()
        else:
            self._reserve_checkpoint()

    def finalize(self) -> None:
        self.release_retry_reservation()

    def _reserve_checkpoint(self) -> None:
        """Atomically reserve a new checkpoint without overwriting retained work."""
        self._resolve_output()
        if self._reservation_owned:
            msg = (
                "ManifestCheckpointStage setup was called again while this stage "
                "still owns an incomplete reservation; reset_for_retry() is required"
            )
            raise RuntimeError(msg)
        marker_path = f"{self._path}._COMPLETE"
        if self._fs.exists(marker_path):
            msg = (
                "ManifestCheckpointStage refuses to create a checkpoint beside an "
                f"existing completion marker at {self.output_path!r}"
            )
            raise FileExistsError(msg)
        owner_path = self._retry_owner_path()
        try:
            with self._fs.open(owner_path, "xb") as owner:
                owner.write(json.dumps({"token": self._reservation_token}).encode("utf-8"))
        except FileExistsError as exc:
            msg = (
                "ManifestCheckpointStage refuses to replace an existing retry "
                f"ownership record at {self.output_path!r}"
            )
            raise FileExistsError(msg) from exc
        try:
            with self._fs.open(self._path, "xb"):
                pass
        except FileExistsError as exc:
            self._remove_retry_owner_if_owned()
            msg = f"ManifestCheckpointStage refuses to overwrite an existing checkpoint at {self.output_path!r}"
            raise FileExistsError(msg) from exc
        except OSError:
            self._remove_retry_owner_if_owned()
            raise
        try:
            stat = os.stat(self._path)
        except OSError:
            # The exclusive create above belongs to this stage instance. If its
            # identity cannot be recorded, remove only that empty reservation
            # and fail rather than creating an unprovable retry owner.
            self._fs.rm(self._path)
            self._remove_retry_owner_if_owned()
            raise
        self._reservation_identity = (stat.st_dev, stat.st_ino, stat.st_ctime_ns)
        self._reservation_owned = True
        self._checkpoint_rows_written = 0
        self._checkpoint_bytes_written = 0
        try:
            self._write_retry_owner(stat)
        except OSError:
            # The path is still the empty reservation whose identity was just
            # recorded. Remove it and its token rather than leave state that a
            # driver-side retry cannot prove.
            current = os.stat(self._path)
            current_identity = (
                current.st_dev,
                current.st_ino,
                current.st_ctime_ns,
            )
            if current_identity == self._reservation_identity and current.st_size == 0:
                self._fs.rm(self._path)
            self._remove_retry_owner_if_owned()
            self._reset_retry_state()
            raise
        logger.info(f"ManifestCheckpointStage: writing metadata to {self.output_path}")

    def reset_for_retry(self) -> None:
        """Remove only this instance's incomplete reservation before an automatic retry."""
        self._resolve_output()
        marker_path = f"{self._path}._COMPLETE"
        if self._fs.exists(marker_path):
            msg = (
                "ManifestCheckpointStage refuses retry reset because a completion "
                f"marker exists at {self.output_path!r}"
            )
            raise FileExistsError(msg)
        owner = self._read_retry_owner()
        if owner is None or owner.get("token") != self._reservation_token:
            if self._fs.exists(self._path) or owner is not None:
                msg = (
                    "ManifestCheckpointStage refuses retry reset of a checkpoint "
                    f"it did not reserve for this run at {self.output_path!r}"
                )
                raise FileExistsError(msg)
            self._reset_retry_state()
            return
        if self._fs.exists(self._path):
            try:
                stat = os.stat(self._path)
            except OSError as exc:
                msg = f"ManifestCheckpointStage could not verify its retry reservation at {self.output_path!r}"
                raise RuntimeError(msg) from exc
            # Match shared-file verification: device numbers are local to each host.
            identity = (stat.st_ino, stat.st_ctime_ns)
            recorded_identity = (
                owner.get("st_ino"),
                owner.get("st_ctime_ns"),
            )
            if identity != recorded_identity or stat.st_size != owner.get("st_size"):
                msg = (
                    "ManifestCheckpointStage refuses retry reset because the checkpoint "
                    f"at {self.output_path!r} is no longer its exact reservation"
                )
                raise FileExistsError(msg)
            self._fs.rm(self._path)
        self._remove_retry_owner_if_owned()
        self._reset_retry_state()

    def release_retry_reservation(self) -> None:
        """Publish completion only after verifying this run's durable reservation."""
        self._resolve_output()
        marker_path = f"{self._path}._COMPLETE"
        if self._fs.exists(marker_path):
            with self._fs.open(marker_path, encoding="utf-8") as marker:
                if json.load(marker).get("token") == self._reservation_token:
                    return
            msg = f"ManifestCheckpointStage refuses to replace completion marker at {self.output_path!r}"
            raise FileExistsError(msg)
        self._verify_reservation()
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=os.path.dirname(self._path),
            prefix=os.path.basename(marker_path) + ".",
            delete=False,
        ) as marker:
            temporary_path = marker.name
            try:
                json.dump(self._read_retry_owner(), marker)
                marker.flush()
                os.fsync(marker.fileno())
                os.link(temporary_path, marker_path)
            finally:
                os.unlink(temporary_path)
        self._remove_retry_owner_if_owned()
        self._reservation_owned = False
        self._reservation_identity = None

    def _retry_owner_path(self) -> str:
        return f"{self._path}._RETRY_OWNER"

    def _read_retry_owner(self) -> dict[str, Any] | None:
        owner_path = self._retry_owner_path()
        if not self._fs.exists(owner_path):
            return None
        try:
            with self._fs.open(owner_path, "rb") as owner:
                value = json.loads(owner.read().decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _write_retry_owner(self, stat: os.stat_result) -> None:
        payload = {
            "token": self._reservation_token,
            "st_dev": stat.st_dev,
            "st_ino": stat.st_ino,
            "st_ctime_ns": stat.st_ctime_ns,
            "st_size": stat.st_size,
        }
        with self._fs.open(self._retry_owner_path(), "wb") as owner:
            owner.write(json.dumps(payload, sort_keys=True).encode("utf-8"))

    def _remove_retry_owner_if_owned(self) -> None:
        owner = self._read_retry_owner()
        if owner is None or owner.get("token") != self._reservation_token:
            return
        self._fs.rm(self._retry_owner_path())

    def _reset_retry_state(self) -> None:
        self._driver_prepared = False
        self._reservation_owned = False
        self._reservation_identity = None
        self._checkpoint_rows_written = 0
        self._checkpoint_bytes_written = 0
        self._custom_metrics = {}

    def process(self, task: AudioTask) -> AudioTask:
        if not self._reservation_owned:
            msg = "ManifestCheckpointStage cannot write without an owned setup reservation"
            raise RuntimeError(msg)
        self._verify_reservation()
        t0 = time.perf_counter()
        row = (json.dumps(task.data, ensure_ascii=False) + "\n").encode("utf-8")
        with self._fs.open(self._path, "ab") as f:
            f.write(row)
        self._checkpoint_rows_written += 1
        self._checkpoint_bytes_written += len(row)
        stat = os.stat(self._path)
        self._reservation_identity = (stat.st_dev, stat.st_ino, stat.st_ctime_ns)
        self._write_retry_owner(stat)
        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "checkpoint_rows_written": 1,
                "checkpoint_bytes_written": len(row),
            }
        )
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
                requires_serializable_input=True,
                per_row_independent=True,
            ),
        )
