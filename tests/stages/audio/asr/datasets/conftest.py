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

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from datasets import Dataset

NUM_REALISTIC_ROWS = 100
INPUT_SAMPLE_RATE = 8000
OUTPUT_SAMPLE_RATE = 16000


@pytest.fixture
def indicvoices_raw_dataset(tmp_path: Path) -> tuple[Path, int]:
    """Create a tiny on-disk HF dataset shaped like the raw IndicVoices split."""
    raw_root = tmp_path / "raw"
    audio_dir = tmp_path / "source_audio"
    valid_dir = raw_root / "valid"
    audio_dir.mkdir(parents=True)

    filepaths = []
    for i in range(NUM_REALISTIC_ROWS):
        samples = np.linspace(0.0, 1.0, INPUT_SAMPLE_RATE // 20, endpoint=False, dtype=np.float32)
        tone = 0.1 * np.sin(2 * np.pi * (220 + i) * samples)
        stereo = np.stack([tone, tone * 0.5], axis=1)
        path = audio_dir / f"sample_{i}.wav"
        sf.write(path, stereo, INPUT_SAMPLE_RATE, subtype="PCM_16")
        filepaths.append(str(path))

    dataset = Dataset.from_dict(
        {
            "audio_filepath": filepaths,
            "text": ["ગુજરાતી વાક્ય" for _ in range(NUM_REALISTIC_ROWS)],
            "duration": [0.05] * NUM_REALISTIC_ROWS,
            "lang": ["gu"] * NUM_REALISTIC_ROWS,
            "speaker_id": [f"speaker_{i % 3}" for i in range(NUM_REALISTIC_ROWS)],
            "gender": ["Female" if i % 2 else "Male" for i in range(NUM_REALISTIC_ROWS)],
            "age_group": ["30-45"] * NUM_REALISTIC_ROWS,
            "scenario": ["Extempore"] * NUM_REALISTIC_ROWS,
            "task_name": ["Unit Test"] * NUM_REALISTIC_ROWS,
            "state": ["Gujarat"] * NUM_REALISTIC_ROWS,
            "district": ["Ahmedabad"] * NUM_REALISTIC_ROWS,
            "normalized": ["ગુજરાતી વાક્ય" for _ in range(NUM_REALISTIC_ROWS)],
        }
    )
    dataset.save_to_disk(str(valid_dir))
    return raw_root, NUM_REALISTIC_ROWS
