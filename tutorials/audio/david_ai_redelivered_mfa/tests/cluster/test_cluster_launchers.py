# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUBMIT = ROOT / "cluster" / "run_multinode.sh"
NODE = ROOT / "cluster" / "run_node.sh"


def test_cluster_launchers_are_executable_and_do_not_copy_data() -> None:
    for script in (SUBMIT, NODE):
        text = script.read_text(encoding="utf-8")
        assert os.access(script, os.X_OK)
        assert "scp " not in text
        assert "rsync " not in text
        assert "--export ALL" not in text
        assert "--export=ALL" not in text


def test_multinode_launcher_uses_explicit_slurm_array_sharding() -> None:
    submit = SUBMIT.read_text(encoding="utf-8")
    node = NODE.read_text(encoding="utf-8")

    assert '--array "0-$((NUM_NODES - 1))%$MAX_CONCURRENT_NODES"' in submit
    assert '--export "$EXPORTS"' in submit
    assert 'SHARD_COUNT="$NUM_NODES"' not in submit
    assert 'EXPORTS+=",SHARD_COUNT=$NUM_NODES"' in submit
    assert 'EXPORTS+=",SESSIONS_FILE=$SESSIONS_FILE"' in submit
    assert "CONTAINER_MOUNTS_B64" in submit
    assert "CONTAINER_MOUNTS=$CONTAINER_MOUNTS" not in submit
    assert 'SHARD_INDEX="$SLURM_ARRAY_TASK_ID"' in node
    assert 'SHARD_COUNT="$SHARD_COUNT"' in node
    assert 'SESSIONS_FILE="$SESSIONS_FILE"' in node


def test_both_pipeline_variants_are_supported() -> None:
    submit = SUBMIT.read_text(encoding="utf-8")
    node = NODE.read_text(encoding="utf-8")

    assert "opus | wav" in submit
    assert 'PIPELINE_DIR="$TUTORIAL_ROOT/$VARIANT"' in node


def test_mfa_scratch_and_model_copies_are_isolated_per_shard_and_worker() -> None:
    node = NODE.read_text(encoding="utf-8")
    assert "${SLURM_JOB_ID}_${SHARD_INDEX}" in node
    assert 'NODE_MFA_ROOT_DIR="$RAM_DIR/model_source"' in node
    assert 'export MFA_ROOT_DIR="$NODE_MFA_ROOT_DIR"' in node
    assert "stage_model" in node

    for variant in ("opus", "wav"):
        ram_session = (ROOT / variant / "david_ai_ram_session.py").read_text(encoding="utf-8")
        common = (ROOT / variant / "david_ai_common.py").read_text(encoding="utf-8")
        aligner = (ROOT / variant / "david_ai_mfa_align.py").read_text(encoding="utf-8")

        assert 'ram_dir / "mfa_workers" / f"worker_{os.getpid()}"' in ram_session
        assert "temp_parent=temp_parent / session_id" in ram_session
        assert 'models_dir = worker_dir / "models"' in common
        assert "shutil.copy2(mfa_dict, local_dict)" in common
        assert "shutil.copytree(acoustic_src, local_acoustic)" in common
        assert 'env["MFA_ROOT_DIR"] = str(mfa_root)' in common
        assert "mfa_subprocess_env(temp_root=temp_root, mfa_root=mfa_root)" in aligner
