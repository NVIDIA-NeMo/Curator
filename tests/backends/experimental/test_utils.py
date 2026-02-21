# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import ray
from pytest import LogCaptureFixture

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.backends.experimental import utils
from nemo_curator.backends.experimental.utils import (
    RayStageSpecKeys,
    execute_setup_on_node,
    get_head_node_id,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources

if TYPE_CHECKING:
    from nemo_curator.tasks import Task


@contextmanager
def _reset_head_node_cache_context() -> Iterator[None]:
    original_value = utils._HEAD_NODE_ID_CACHE
    utils._HEAD_NODE_ID_CACHE = None
    try:
        yield
    finally:
        utils._HEAD_NODE_ID_CACHE = original_value


@pytest.fixture
def reset_head_node_cache() -> Iterator[None]:
    with _reset_head_node_cache_context():
        yield


class TestExecuteSetupOnNode:
    """Test class for execute_setup_on_node function."""

    def test_execute_setup_on_node_with_two_stages(
        self,
        shared_ray_client: None,
        tmp_path: Path,
        caplog: LogCaptureFixture,
    ):
        """Test execute_setup_on_node with two stages on the Ray cluster."""

        class MockStage1(ProcessingStage):
            name = "mock_stage_1"
            resources = Resources(cpus=1.0, gpus=0.0)

            def process(self, task: "Task") -> "Task":
                return task

            def setup_on_node(
                self, node_info: NodeInfo | None = None, worker_metadata: WorkerMetadata | None = None
            ) -> None:
                # Write a file to record this call
                node_id = node_info.node_id if node_info else "unknown"
                worker_id = worker_metadata.worker_id if worker_metadata else "unknown"
                filename = f"{self.name}_{uuid.uuid4()}.txt"
                filepath = tmp_path / filename
                with open(filepath, "w") as f:
                    f.write(f"{node_id},{worker_id}\n")

        stage1 = MockStage1()
        stage2 = MockStage1().with_(name="mock_stage_2", resources=Resources(cpus=0.5, gpus=0.0))

        # Test
        execute_setup_on_node([stage1, stage2])

        # Check the files written to the temp directory
        # Verify that NodeInfo and WorkerMetadata were passed correctly
        for stage_name in ["mock_stage_1", "mock_stage_2"]:
            stage_files = list(tmp_path.glob(f"{stage_name}_*.txt"))
            assert len(stage_files) == len(ray.nodes()), (
                f"Expected {len(ray.nodes())} calls to setup_on_node for {stage_name}, got {len(stage_files)}"
            )
            node_ids = set()
            for file_path in stage_files:
                content = file_path.read_text().strip()
                node_id, worker_id = content.split(",")
                assert worker_id == "", f"{stage_name} Worker ID should be empty string, got '{worker_id}'"
                node_ids.add(node_id)
            assert len(node_ids) == len(ray.nodes()), (
                f"Expected {len(ray.nodes())} different node IDs for {stage_name}, got {node_ids}"
            )
            assert node_ids == {node["NodeID"] for node in ray.nodes()}, (
                f"Expected node IDs to be the same as the Ray nodes, got {node_ids}"
            )

        # Check that there are exactly two log records that start with "Executing setup on node" and end with "for 2 stages"
        matching_logs = [
            record.message
            for record in caplog.records
            if record.message.startswith("Executing setup on node") and record.message.endswith("for 2 stages")
        ]
        # TODO: When we add a cluster then we should check the value of len(ray.nodes()) too
        assert len(matching_logs) == len(ray.nodes()), (
            f"Expected {len(ray.nodes())} logs for setup on node for 2 stages, got {len(matching_logs)}: {matching_logs}"
        )

    def test_execute_setup_on_node_ignore_head_node(
        self,
        shared_ray_client: None,
        tmp_path: Path,
        caplog: LogCaptureFixture,
        reset_head_node_cache: None,
    ):
        """Test execute_setup_on_node with ignore_head_node=True to skip head node."""

        class MockStage1(ProcessingStage):
            name = "mock_stage_ignore_head"
            resources = Resources(cpus=1.0, gpus=0.0)

            def process(self, task: "Task") -> "Task":
                return task

            def setup_on_node(
                self, node_info: NodeInfo | None = None, worker_metadata: WorkerMetadata | None = None
            ) -> None:
                # Write a file to record this call
                node_id = node_info.node_id if node_info else "unknown"
                worker_id = worker_metadata.worker_id if worker_metadata else "unknown"
                filename = f"{self.name}_{uuid.uuid4()}.txt"
                filepath = tmp_path / filename
                with open(filepath, "w") as f:
                    f.write(f"{node_id},{worker_id}\n")

        stage = MockStage1()

        # Test with ignore_head_node=True
        execute_setup_on_node([stage], ignore_head_node=True)

        # Verify the cache variable is set directly (not using the lazy function)
        assert utils._HEAD_NODE_ID_CACHE is not None, "_HEAD_NODE_ID_CACHE should be set after execute_setup_on_node"

        # Verify it matches the actual head node in the cluster
        expected_head_node_id = None
        for node in ray.nodes():
            if "node:__internal_head__" in node.get("Resources", {}):
                expected_head_node_id = node["NodeID"]
                break

        assert expected_head_node_id is not None, "Expected head node ID should be set"
        assert expected_head_node_id == utils._HEAD_NODE_ID_CACHE, (
            f"_HEAD_NODE_ID_CACHE should be {expected_head_node_id}, got {utils._HEAD_NODE_ID_CACHE}"
        )

        # Check the files written to the temp directory
        stage_files = list(tmp_path.glob(f"{stage.name}_*.txt"))
        expected_calls = len(ray.nodes()) - (1 if expected_head_node_id else 0)
        assert len(stage_files) == expected_calls, (
            f"Expected {expected_calls} calls to setup_on_node (excluding head node), got {len(stage_files)}"
        )


class TestGetHeadNodeId:
    def test_lazy_evaluation(
        self,
        shared_ray_client: None,
        reset_head_node_cache: None,
    ):
        """Test that get_head_node_id uses lazy evaluation and caching."""

        # Cache should start cleared by fixture
        assert utils._HEAD_NODE_ID_CACHE is None, "Cache should be cleared before test"

        # First call should compute and cache
        head_node_id_1 = get_head_node_id()

        # Cache should now be set
        assert utils._HEAD_NODE_ID_CACHE is not None, "Cache should be set after first call"

        # Second call should return cached value
        head_node_id_2 = get_head_node_id()

        # Both should be the same
        assert head_node_id_1 == head_node_id_2, "Cached value should match"

        # Verify it's the actual head node
        expected_head_node_id = None
        for node in ray.nodes():
            if "node:__internal_head__" in node.get("Resources", {}):
                expected_head_node_id = node["NodeID"]
                break

        assert expected_head_node_id is not None, "Expected head node ID should be set"
        assert head_node_id_1 == expected_head_node_id, (
            f"get_head_node_id() returned {head_node_id_1}, expected {expected_head_node_id}"
        )


class TestRayStageSpecKeys:
    """Test class for RayStageSpecKeys enum compatibility."""

    def test_enum_membership_compatibility(self):
        """Test that the fixed pattern works across Python versions."""
        # Test data
        valid_keys = ["is_actor_stage", "is_fanout_stage", "is_lsh_stage"]
        invalid_keys = ["invalid_key", "another_bad_key"]

        # Test the fixed pattern - this is what's now used in the adapter
        enum_values = {e.value for e in RayStageSpecKeys}

        # Testing valid keys
        for key in valid_keys:
            result = key not in enum_values
            assert result is False, f"Valid key '{key}' should be found in enum values"

        # Testing invalid keys
        for key in invalid_keys:
            result = key not in enum_values
            assert result is True, f"Invalid key '{key}' should not be found in enum values"
