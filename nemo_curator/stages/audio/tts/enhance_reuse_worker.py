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

"""Out-of-process Resemble-Enhance ("ReUse") worker for ``ResembleEnhanceStage``.

Runs in a SEPARATE interpreter that has ``resemble-enhance`` installed (it pins ``torch==2.1``, which conflicts with
the Curator/NeMo torch), so it imports nothing from Curator. Usage::

    python enhance_reuse_worker.py --manifest items.jsonl --nfe 32 [--sr 44100] [--no-denoise]

``items.jsonl`` holds ``{"in": <wav>, "out": <wav>}`` per line. The model is loaded once per invocation; each item is
isolated (a failure leaves no output file). Pre-staged weights: set ``RESEMBLE_RUN_DIR`` to the ``enhancer_stage2``
directory (otherwise resemble downloads them with git-lfs, which fails on air-gapped nodes).
Prints one JSON result line per item: ``{"in", "out", "ok", "sampling_rate", "num_samples", "error"}``.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    # Running this file puts its directory (the Curator ``tts`` package) on sys.path; drop it so that
    # ``import resemble_enhance`` cannot resolve to a sibling module.
    """Enhance every item of ``--manifest`` and print one JSON result line per item."""
    here = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if p and str(Path(p).resolve()) != here]
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--solver", default="midpoint")
    ap.add_argument("--lambd", type=float, default=0.9)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--sr", type=int, default=0, help="output sample rate (0 = keep the enhancer's native rate)")
    ap.add_argument("--no-denoise", action="store_true")
    a = ap.parse_args()

    import soundfile as sf
    import torch
    import torchaudio.functional as af
    from resemble_enhance.enhancer.inference import denoise, enhance

    run_dir = Path(os.environ["RESEMBLE_RUN_DIR"]) if os.environ.get("RESEMBLE_RUN_DIR") else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = [json.loads(line) for line in Path(a.manifest).read_text().splitlines() if line.strip()]
    ok = 0
    for item in items:
        result = {"in": item["in"], "out": item["out"], "ok": False, "sampling_rate": None, "num_samples": None}
        try:
            data, sr = sf.read(item["in"], dtype="float32", always_2d=True)  # soundfile: no torchcodec needed
            wav = torch.from_numpy(data.mean(axis=1))
            if not a.no_denoise:
                wav, sr = denoise(wav, sr, device, run_dir=run_dir)
            wav, sr = enhance(wav, sr, device, nfe=a.nfe, solver=a.solver, lambd=a.lambd, tau=a.tau, run_dir=run_dir)
            if a.sr and sr != a.sr:
                wav, sr = af.resample(wav, sr, a.sr), a.sr
            out = Path(item["out"])
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".tmp.wav")
            sf.write(str(tmp), wav.detach().cpu().float().numpy(), int(sr), subtype="PCM_16", format="WAV")
            tmp.replace(out)  # atomic: a half-written file never looks done
            result.update(ok=True, sampling_rate=int(sr), num_samples=int(wav.shape[-1]), error=None)
            ok += 1
        except Exception as exc:  # noqa: BLE001  (isolate per-item failures)
            result["error"] = repr(exc)[:300]
            if device == "cuda":
                torch.cuda.empty_cache()
        print(json.dumps(result), flush=True)
    return 0 if ok or not items else 1


if __name__ == "__main__":
    sys.exit(main())
