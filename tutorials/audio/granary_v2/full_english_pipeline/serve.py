# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve the full pipeline's text model on its separate GPU allocation."""

import argparse
import signal
from pathlib import Path
from threading import Event

from huggingface_hub import snapshot_download
from hydra.utils import instantiate
from omegaconf import OmegaConf

from nemo_curator.core.client import RayClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", help="Prepared snapshot of the YAML's pinned model revision")
    parser.add_argument("--cache-root", default="/cache/compilation")
    args = parser.parse_args()
    cfg = OmegaConf.load("tutorials/audio/granary_v2/full_english_pipeline/pipeline.yaml")
    cfg.cache_root = args.cache_root
    cfg.model_name = str(cfg.model_name)
    if args.model_dir:
        cfg.model_identifier = args.model_dir
    elif not Path(cfg.model_identifier).is_dir():
        cfg.model_identifier = snapshot_download(
            repo_id=cfg.model_identifier, revision=cfg.inference_server.models[0].engine_kwargs.revision
        )
    stopped = Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    with (
        RayClient(num_cpus=128, num_gpus=8, include_dashboard=False),
        instantiate(cfg.inference_server) as server,
    ):
        print(f"Ready: {server.endpoint}", flush=True)
        stopped.wait()


if __name__ == "__main__":
    main()
