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

from david_ai_common import append_mfa_g2p_args, mfa_subprocess_env


def test_append_mfa_g2p_args_adds_flag():
    cmd = ["mfa", "align"]
    append_mfa_g2p_args(cmd, g2p_path="/tmp/english_us_arpa.zip")
    assert cmd == ["mfa", "align", "--g2p_model_path", "/tmp/english_us_arpa.zip"]


def test_append_mfa_g2p_args_skips_empty():
    cmd = ["mfa", "align"]
    append_mfa_g2p_args(cmd, g2p_path=None)
    assert cmd == ["mfa", "align"]


def test_mfa_subprocess_env_strips_container_pythonpath(tmp_path):
    fake_env = tmp_path / "env"
    (fake_env / "lib").mkdir(parents=True)
    os.environ["PYTHONPATH"] = "/tmp/curator_pkg:/opt/Export-Deploy:/opt/venv/lib/python3.12/site-packages:"
    os.environ["MFA_ENV"] = str(fake_env)
    env = mfa_subprocess_env(temp_root=tmp_path / "align", mfa_root=tmp_path / "mfa_root")
    assert "/opt/venv" not in env.get("PYTHONPATH", "")
    assert env["PYTHONPATH"] == "/tmp/curator_pkg"
    assert env["LD_LIBRARY_PATH"].startswith(str(fake_env / "lib"))
    del os.environ["PYTHONPATH"]
    del os.environ["MFA_ENV"]
