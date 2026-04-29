"""Background GPU keep-alive to avoid Slurm under-utilization watchdogs.

Some cluster policies (e.g. draco-oci) cancel jobs whose GPUs are idle
for too long.  In a multi-stage pipeline, CPU-only stages (SED post,
segment fan-out, SCOTCH clustering) leave the GPU at 0% for tens of
minutes, which trips the watchdog even though the job is making progress.

This script runs a negligible matmul on every visible GPU every few
seconds.  It exists to keep ``nvidia-smi`` reporting nonzero compute
activity; it does not contend meaningfully for resources.

Run as a background process for the lifetime of the srun step::

    python tutorials/audio/hifi_pipeline/gpu_keepalive.py &
    KP=$!
    trap "kill -TERM $KP 2>/dev/null || true" EXIT

Implementation notes:
    * ~3 microseconds per cycle on A100; overhead during GPU-busy stages
      is unmeasurable.
    * Allocates a 512x512 fp32 tensor per GPU (1 MB each).
    * Catches SIGTERM/SIGINT so trap-driven cleanup is graceful.
"""
from __future__ import annotations

import os
import signal
import sys
import time

# Tunables (env-overridable so we don't need a code change to retune).
# Defaults target ~10% sustained GPU utilization on A100: large enough
# matmul + short sleep so nvidia-smi consistently samples nonzero usage,
# which is what cluster under-utilization watchdogs check.
_MATRIX_DIM = int(os.environ.get("GPU_KEEPALIVE_DIM", "4096"))
_BUSY_SECONDS = float(os.environ.get("GPU_KEEPALIVE_BUSY_S", "0.2"))
_IDLE_SECONDS = float(os.environ.get("GPU_KEEPALIVE_IDLE_S", "1.5"))


def main() -> None:
    try:
        import torch
    except ImportError:
        # No torch -> nothing to keep alive; exit silently.
        sys.exit(0)

    if not torch.cuda.is_available():
        sys.exit(0)

    n = torch.cuda.device_count()
    matrices = [
        torch.randn(_MATRIX_DIM, _MATRIX_DIM, device=f"cuda:{i}")
        for i in range(n)
    ]
    print(
        f"[gpu_keepalive] running on {n} GPU(s), "
        f"{_MATRIX_DIM}x{_MATRIX_DIM} fp32, "
        f"busy={_BUSY_SECONDS}s idle={_IDLE_SECONDS}s "
        f"(~{100 * _BUSY_SECONDS / (_BUSY_SECONDS + _IDLE_SECONDS):.0f}% duty)",
        flush=True,
    )

    def _stop(_sig: int, _frame: object) -> None:  # noqa: ARG001
        print("[gpu_keepalive] stop requested", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    # Sustained busy phase so nvidia-smi's per-second sample window
    # consistently captures nonzero compute, then idle to share with
    # real workloads.
    while True:
        deadline = time.time() + _BUSY_SECONDS
        while time.time() < deadline:
            for m in matrices:
                m.copy_((m @ m.T) * 1e-6 + 1.0)
        torch.cuda.synchronize()
        time.sleep(_IDLE_SECONDS)


if __name__ == "__main__":
    main()
