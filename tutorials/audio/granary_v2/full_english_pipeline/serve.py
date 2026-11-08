# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start the separate text server with the tutorial's tested worker environment."""

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from threading import Event

import ray
from huggingface_hub import snapshot_download
from hydra.utils import instantiate
from omegaconf import OmegaConf
from packaging.utils import canonicalize_name

from nemo_curator.core.client import RayClient
from nemo_curator.core.serve import DynamoVLLMModelConfig, InferenceServer
from nemo_curator.core.serve.dynamo.vllm import dynamo_runtime_env, ensure_actor_overrides_on_all_nodes

TUTORIAL = Path(__file__).resolve().parent


def inspect_worker() -> dict:
    """Inspect the interpreter used by Dynamo's child, outside Ray's import path."""
    code = (
        "import importlib.metadata,json,sys; "
        "print(json.dumps({'python':sys.version,'packages':"
        "{d.metadata['Name']:importlib.metadata.version(d.metadata['Name']) "
        "for d in importlib.metadata.distributions() if d.metadata['Name']}}))"
    )
    return json.loads(subprocess.check_output([sys.executable, "-I", "-c", code], text=True))  # noqa: S603


def check_worker(model: DynamoVLLMModelConfig) -> dict:
    ensure_actor_overrides_on_all_nodes()
    actual = ray.get(
        ray.remote(num_cpus=1, runtime_env=dynamo_runtime_env(model))(inspect_worker).remote(), timeout=900
    )
    expected = {}
    for line in (TUTORIAL / "text-worker-constraints.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==", 1)
            expected[canonicalize_name(name)] = version
    versions = {canonicalize_name(name): version for name, version in actual["packages"].items()}
    if versions != expected or not actual["python"].startswith("3.13.16 "):
        changed = {
            name: (expected.get(name), versions.get(name))
            for name in expected.keys() | versions.keys()
            if expected.get(name) != versions.get(name)
        }
        message = f"Text worker differs from the tested environment: Python {actual['python']}; packages {changed}"
        raise RuntimeError(message)
    return actual


@contextmanager
def text_server(model_dir: str | None, cache_root: str) -> Iterator[tuple[InferenceServer, dict]]:
    cfg = OmegaConf.load(TUTORIAL / "pipeline.yaml")
    cfg.cache_root = cache_root
    cfg.model_name = str(cfg.model_name)
    cfg.model_identifier = model_dir or snapshot_download(
        repo_id=cfg.model_identifier, revision=cfg.inference_server.models[0].engine_kwargs.revision
    )
    client = RayClient(num_cpus=128, num_gpus=8, include_dashboard=False)
    try:
        client.start()
        ray.init(address=os.environ["RAY_ADDRESS"])
        inventory = check_worker(instantiate(cfg.inference_server.models[0]))
        ray.shutdown()
        with instantiate(cfg.inference_server) as server:
            yield server, inventory
    finally:
        ray.shutdown()
        client.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", help="Prepared snapshot of the YAML's pinned model revision")
    parser.add_argument("--cache-root", default="/cache/compilation")
    parser.add_argument("--endpoint-file", default="/output/endpoint.txt")
    args = parser.parse_args()
    endpoint = Path(args.endpoint_file)
    if endpoint.exists():
        raise FileExistsError(endpoint)
    with text_server(args.model_dir, args.cache_root) as (server, _inventory):
        with endpoint.open("x") as stream:
            stream.write(server.endpoint + "\n")
        print(f"Ready: {server.endpoint}. Ctrl-C stops serving.", flush=True)
        with suppress(KeyboardInterrupt):
            Event().wait()


if __name__ == "__main__":
    main()
