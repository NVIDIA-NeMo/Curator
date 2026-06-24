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

"""Compatibility patch for Dynamo's embedding worker in this benchmark.

Dynamo 1.3.0.dev20260615 leaves ``PoolingParams.task`` unset in its
``EmbeddingWorkerHandler``. vLLM then defaults embeddinggemma requests to
``token_embed`` because the model supports token-level embeddings. The OpenAI
embeddings endpoint should request ``embed`` so each row returns one pooled
vector. This patch is enabled only when the benchmark sets the env var below.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _patch_dynamo_embedding_handler() -> None:
    if os.environ.get("NEMO_CURATOR_PATCH_DYNAMO_EMBED_POOLING_TASK") != "1":
        return

    spec = importlib.util.find_spec("dynamo.vllm.handlers")
    if spec is None or spec.origin is None:
        print("NeMo Curator: could not locate dynamo.vllm.handlers for embedding patch", file=sys.stderr)
        return

    handlers_path = Path(spec.origin)
    source = handlers_path.read_text()
    original = "        pooling_params = PoolingParams()\n"
    patched = '        pooling_params = PoolingParams(task="embed", dimensions=dimensions)\n'
    if original in source:
        source = source.replace(original, patched, 1)
        print(f"NeMo Curator: patched Dynamo embedding PoolingParams in {handlers_path}", file=sys.stderr)
    elif patched not in source:
        print(
            f"NeMo Curator: Dynamo embedding patch did not match {handlers_path}",
            file=sys.stderr,
        )
        return

    if os.environ.get("NEMO_CURATOR_PATCH_DYNAMO_EMBED_NO_ABORT_MONITOR") == "1":
        original = """            final_output = None
            async with self._abort_monitor(context, request_id):
                async for out in self.engine_client.encode(
                    prompt=encode_arg,
                    pooling_params=pooling_params,
                    request_id=request_id,
                ):
                    final_output = out
"""
        patched = """            final_output = None
            async for out in self.engine_client.encode(
                prompt=encode_arg,
                pooling_params=pooling_params,
                request_id=request_id,
            ):
                final_output = out
"""
        if original in source:
            source = source.replace(original, patched, 1)
            print(
                f"NeMo Curator: patched Dynamo embedding abort monitor in {handlers_path}",
                file=sys.stderr,
            )
        elif patched not in source:
            print(
                f"NeMo Curator: Dynamo embedding abort-monitor patch did not match {handlers_path}",
                file=sys.stderr,
            )
            return

    handlers_path.write_text(source)


_patch_dynamo_embedding_handler()
