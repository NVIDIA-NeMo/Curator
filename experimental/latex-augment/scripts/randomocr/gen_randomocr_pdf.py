# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import tqdm
from latex_augment.randomocr import generate_sample

num_docs = 1000
seed = 1234

# script = "ascii"
script = "english"
# script = "latin"
# script = "greek"
# script = "chinese"
# script = "japanese"
# script = "korean"
outdir = "randomocr_" + script


def gen(sample_id):
    _, png, label, latex, text, pdf = generate_sample(script, sample_id)
    with open(f"{outdir}/{sample_id:04d}.txt", "w") as f:
        f.write(text)
    with open(f"{outdir}/{sample_id:04d}.tex", "wb") as f:
        f.write(latex)
    with open(f"{outdir}/{sample_id:04d}.pdf", "wb") as f:
        f.write(pdf)
    with open(f"{outdir}/{sample_id:04d}.png", "wb") as f:
        f.write(png)
    with open(f"{outdir}/{sample_id:04d}.doclaynet.json", "w") as f:
        json.dump(label, f)


# gen(42)
shutil.rmtree(outdir, ignore_errors=True)
os.makedirs(outdir, exist_ok=True)
# list(tqdm.tqdm(map(gen, range(num_docs)), total=num_docs))
with ThreadPoolExecutor() as executor:
    list(tqdm.tqdm(executor.map(gen, range(num_docs)), total=num_docs))
