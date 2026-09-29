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

import subprocess
import sys
import textwrap


def test_vad_stage_import_does_not_load_nemo_or_silero() -> None:
    script = textwrap.dedent(
        """
        import builtins
        import sys

        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "silero_vad" or name.startswith("nemo.collections"):
                raise AssertionError(f"optional dependency imported eagerly: {name}")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = guarded_import
        from nemo_curator.stages.audio.segmentation.vad_segmentation import VADSegmentationStage

        assert VADSegmentationStage.__name__ == "VADSegmentationStage"
        assert "silero_vad" not in sys.modules
        assert not any(name.startswith("nemo.collections") for name in sys.modules)
        """
    )

    subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
