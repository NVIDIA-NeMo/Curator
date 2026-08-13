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

import sys
from pathlib import Path

from pytest import MonkeyPatch

BENCHMARKING_DIR = Path(__file__).resolve().parents[2] / "benchmarking"
sys.path.insert(0, str(BENCHMARKING_DIR))
from runner.session import Session  # noqa: E402
from runner.utils import merge_config_files, remove_disabled_blocks, resolve_env_vars  # noqa: E402


def test_sortformer_nightly_inherits_eight_gpus(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("SLACK_CHANNEL_ID", "test")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "test")
    config = merge_config_files(
        [root / "benchmarking/nightly-benchmark.yaml", root / "benchmarking/nightly-data-setup.yaml"]
    )
    sortformer_entry = next(entry for entry in config["entries"] if entry["name"] == "audio_sortformer_xenna")
    assert sortformer_entry["enabled"] is True
    config = resolve_env_vars(remove_disabled_blocks(config), strict=True)
    session = Session.from_dict(config, entries_exact=["audio_sortformer_xenna"])

    (entry,) = session.entries
    assert entry.ray == {"num_cpus": 128, "num_gpus": 8, "enable_object_spilling": False}
    assert entry.timeout_s == 1800
    command = entry.get_command_to_run(tmp_path / entry.name, session.path_resolver, session.dataset_resolver)
    assert "--repeat-factor" not in command
    assert "--scratch-output-path" in command
    assert "--gpu-stage-num-workers=8" in command
    assert "--chunk-len=6" in command
    assert "--chunk-left-context=1" in command
    assert "--chunk-right-context=7" in command
    assert "--fifo-len=188" in command
    assert "--spkcache-update-period=144" in command
    assert "--spkcache-len=188" in command
    assert "--executor=xenna" in command
    assert "{" not in command
    assert entry.requirements["num_input_rows"]["exact_value"] == 34
    assert entry.requirements["num_output_rows"]["exact_value"] == 34
    assert entry.requirements["num_tasks_with_segments"]["exact_value"] == 34
    assert entry.requirements["stage_execution_coverage_ratio"]["exact_value"] == 1.0
    assert entry.requirements["total_audio_duration_hours"]["min_value"] == 18.7
    assert entry.requirements["throughput_audio_hours_per_hour"]["min_value"] == 1.0

    setup = next(item for item in session.data_setups if item.name == "audio_sortformer_ami_sdm")
    setup_command = setup.get_command_to_run(tmp_path / "setup", session.path_resolver, session.dataset_resolver)
    assert "prepare_audio_sortformer_data.py" in setup_command
    assert "diar_streaming_sortformer_4spk-v2.1.nemo" in setup_command
    assert "{" not in setup_command
