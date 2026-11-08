# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve the full pipeline's text model on its separate GPU allocation."""

import argparse
import signal
from threading import Event

from omegaconf import OmegaConf

from nemo_curator.config.run import create_inference_server_from_yaml
from nemo_curator.core.client import RayClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", help="Prepared snapshot of the YAML's pinned model revision")
    parser.add_argument("--cache-root", default="/cache/compilation")
    args = parser.parse_args()
    cfg = OmegaConf.load("tutorials/audio/granary_v2/full_english_pipeline/pipeline.yaml")
    cfg.cache_root = args.cache_root
    if args.model_dir:
        cfg.model_name = str(cfg.model_name)
        cfg.model_identifier = args.model_dir
    stopped = Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    with (
        RayClient(num_cpus=128, num_gpus=8, include_dashboard=False),
        create_inference_server_from_yaml(cfg) as server,
    ):
        print(f"Ready: {server.endpoint}", flush=True)
        stopped.wait()


if __name__ == "__main__":
    main()
