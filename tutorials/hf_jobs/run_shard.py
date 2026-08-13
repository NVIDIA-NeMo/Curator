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

# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "nemo-curator==1.3.0",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""HF Jobs shim for `tutorials/slurm/array_pipeline.py`.

This file is the Hugging Face Jobs analog of `submit_array.sh`: all
scheduler-side plumbing lives here, and the tutorial pipeline runs unmodified.

`hf jobs uv run` uploads a single script file, so the sibling
`../slurm/array_pipeline.py` is not present inside the Job container. The shim
therefore downloads it at the release tag matching the pinned `nemo-curator`
dependency above and verifies its sha256 before executing it.

Container quirks handled here (and only here):
  [1] cwd is a FUSE bucket mount; Ray's runtime_env packaging hashes cwd -> chdir to /tmp
  [2] Ray autodetects the cgroup (1 CPU on cpu-basic); the Xenna executor's
      streaming floor needs ~3.5 -> logical overcommit via RayClient(num_cpus=...)
  [3] the mount lacks POSIX rename; Curator writes completion manifests
      atomically (write + rename) -> checkpoint locally, then publish the
      manifest JSONs to the shared mount with plain writes
"""

import hashlib
import os
import runpy
import sys
import tempfile
import urllib.request
from pathlib import Path

# Keep in sync with the nemo-curator version pinned in the header above.
UPSTREAM_REF = "v1.3.0"
ARRAY_PIPELINE_SHA256 = "80870f20a0a617a8e0ead74be0fca0eff7deaf5478b197a41c0a64834c16af12"
ARRAY_PIPELINE_URL = (
    f"https://raw.githubusercontent.com/NVIDIA-NeMo/Curator/{UPSTREAM_REF}/tutorials/slurm/array_pipeline.py"
)

SHARE = Path(os.environ.get("TUTORIAL_SHARE", "/mnt/tutorial"))


def fetch_upstream_pipeline(dest_dir: Path) -> Path:
    dest = dest_dir / "array_pipeline.py"
    with urllib.request.urlopen(ARRAY_PIPELINE_URL, timeout=60) as r:
        data = r.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != ARRAY_PIPELINE_SHA256:
        msg = f"upstream file changed: sha256 {digest} != pinned {ARRAY_PIPELINE_SHA256}"
        raise RuntimeError(msg)
    dest.write_bytes(data)
    print(f"[shim] fetched array_pipeline.py @ {UPSTREAM_REF} sha256-verified", flush=True)
    return dest


def main() -> None:
    shard = os.environ.get("NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX", "?")
    total = os.environ.get("NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS", "?")
    print(f"[shim] shard {shard}/{total} share={SHARE}", flush=True)

    # [1] work from local disk, never from the mount
    work = Path(tempfile.mkdtemp(prefix="curator-tutorial-"))
    ckpt_local = Path(tempfile.mkdtemp(prefix="curator-ckpt-"))
    os.chdir(work)

    # [2] transparent RayClient default override — the tutorial file stays unchanged
    import nemo_curator.core.client as _client

    _original_ray_client = _client.RayClient

    class _JobsRayClient(_original_ray_client):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("num_cpus", int(os.environ.get("HF_JOBS_RAY_NUM_CPUS", "8")))
            kwargs.setdefault("include_dashboard", False)
            super().__init__(*args, **kwargs)

    _client.RayClient = _JobsRayClient

    pipeline_file = fetch_upstream_pipeline(work)

    # [3] checkpoint locally; the pipeline CLI accepts the path, no file changes needed
    sys.argv = [
        "array_pipeline.py",
        "--input-dir",
        str(SHARE / "input"),
        "--input-file-type",
        os.environ.get("INPUT_FILE_TYPE", "jsonl"),
        "--output-dir",
        str(SHARE / "out"),
        "--output-file-type",
        os.environ.get("OUTPUT_FILE_TYPE", "jsonl"),
        "--files-per-partition",
        os.environ.get("FILES_PER_PARTITION", "1"),
        "--checkpoint-path",
        str(ckpt_local),
    ]
    runpy.run_path(str(pipeline_file), run_name="__main__")

    # [3 cont.] publish completion manifests to the shared mount (plain writes)
    src = ckpt_local / ".nemo_curator_metadata" / ".slurm_array_completion"
    dst = SHARE / "ckpt" / ".nemo_curator_metadata" / ".slurm_array_completion"
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.json")):
        (dst / f.name).write_bytes(f.read_bytes())
        print(f"[shim] published manifest {f.name}", flush=True)
    print(f"SHIM_OK {shard}", flush=True)


if __name__ == "__main__":
    main()
