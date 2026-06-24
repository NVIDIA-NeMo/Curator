import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest
import yaml

from nemo_curator.tasks import DocumentBatch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "benchmarking" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

spec = importlib.util.spec_from_file_location(
    "embedding_generation_benchmark",
    SCRIPTS_DIR / "embedding_generation_benchmark.py",
)
assert spec.loader is not None
embedding_generation_benchmark = importlib.util.module_from_spec(spec)
assert isinstance(embedding_generation_benchmark, ModuleType)
spec.loader.exec_module(embedding_generation_benchmark)


def test_endpoint_embedding_client_stage_batches_rows_and_preserves_order() -> None:
    in_flight = 0
    max_seen = 0
    calls: list[dict] = []

    class FakeEmbeddings:
        async def create(
            self,
            *,
            model: str,
            input: list[str],  # noqa: A002
            encoding_format: str,
            timeout: float,  # noqa: ASYNC109
            extra_body: dict[str, int] | None = None,
        ) -> SimpleNamespace:
            nonlocal in_flight, max_seen
            calls.append(
                {
                    "model": model,
                    "input": input,
                    "encoding_format": encoding_format,
                    "timeout": timeout,
                    "extra_body": extra_body,
                }
            )
            in_flight += 1
            max_seen = max(max_seen, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            # Return out of order to verify the stage restores OpenAI index order.
            data = [
                SimpleNamespace(index=index, embedding=[float(len(text))])
                for index, text in reversed(list(enumerate(input)))
            ]
            return SimpleNamespace(data=data)

    stage = embedding_generation_benchmark.OpenAIEmbeddingClientStage(
        model_identifier="google/embeddinggemma-300m",
        endpoint="http://localhost:8000/v1",
        max_concurrent_requests=2,
        timeout=7.0,
        endpoint_request_batch_size=2,
    )
    stage.client = SimpleNamespace(embeddings=FakeEmbeddings())

    batch = DocumentBatch(
        dataset_name="unit",
        data=pd.DataFrame({"text": ["aa", "b", "cccc"]}),
    )

    output = stage.process(batch)

    assert output.to_pyarrow().column("embeddings").to_pylist() == [[2.0], [1.0], [4.0]]
    assert calls == [
        {
            "model": "google/embeddinggemma-300m",
            "input": ["aa", "b"],
            "encoding_format": "float",
            "timeout": 7.0,
            "extra_body": None,
        },
        {
            "model": "google/embeddinggemma-300m",
            "input": ["cccc"],
            "encoding_format": "float",
            "timeout": 7.0,
            "extra_body": None,
        },
    ]
    assert max_seen == 2
    metrics = stage._consume_custom_metrics()
    assert metrics["embedding_request_count"] == 2.0
    assert metrics["endpoint_max_concurrent_requests"] == 2.0
    assert metrics["endpoint_truncate_prompt_tokens"] == -1.0


def test_endpoint_embedding_client_stage_rejects_missing_response_rows() -> None:
    class FakeEmbeddings:
        async def create(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0])])

    stage = embedding_generation_benchmark.OpenAIEmbeddingClientStage(
        model_identifier="google/embeddinggemma-300m",
        endpoint="http://localhost:8000/v1",
        max_concurrent_requests=1,
        timeout=7.0,
        endpoint_request_batch_size=2,
    )
    stage.client = SimpleNamespace(embeddings=FakeEmbeddings())

    batch = DocumentBatch(
        dataset_name="unit",
        data=pd.DataFrame({"text": ["aa", "b"]}),
    )

    with pytest.raises(RuntimeError, match="returned 1 embeddings for 2 inputs"):
        stage.process(batch)


def test_endpoint_embedding_client_stage_rejects_non_contiguous_response_indexes() -> None:
    class FakeEmbeddings:
        async def create(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=0, embedding=[1.0]),
                    SimpleNamespace(index=0, embedding=[2.0]),
                ]
            )

    stage = embedding_generation_benchmark.OpenAIEmbeddingClientStage(
        model_identifier="google/embeddinggemma-300m",
        endpoint="http://localhost:8000/v1",
        max_concurrent_requests=1,
        timeout=7.0,
        endpoint_request_batch_size=2,
    )
    stage.client = SimpleNamespace(embeddings=FakeEmbeddings())

    batch = DocumentBatch(
        dataset_name="unit",
        data=pd.DataFrame({"text": ["aa", "b"]}),
    )

    with pytest.raises(RuntimeError, match="non-contiguous embedding indexes"):
        stage.process(batch)


def test_endpoint_embedding_stage_uses_cpu_resources_and_fixed_worker_pool() -> None:
    stage = embedding_generation_benchmark._create_endpoint_embedding_stage(
        model_identifier="google/embeddinggemma-300m",
        endpoint="http://localhost:8000/v1",
        client_num_workers=32,
        endpoint_max_concurrent_requests=16,
        endpoint_request_timeout_s=900,
        endpoint_max_retries=3,
        endpoint_retry_base_delay_s=1.0,
        endpoint_truncate_prompt_tokens=-1,
        endpoint_input_format="text",
        endpoint_request_batch_size=8,
        endpoint_encoding_format="float",
        endpoint_max_chars=1500,
        endpoint_client_mode="actor_pool",
        cache_dir=None,
    )

    assert stage.resources.cpus == 1
    assert stage.resources.gpus == 0
    assert stage.num_workers() == 32
    assert stage.max_concurrent_requests == 16
    assert stage.endpoint_truncate_prompt_tokens == -1
    assert stage.endpoint_max_chars == 1500


def test_endpoint_runtime_env_routes_caches_to_writable_model_cache() -> None:
    runtime_env = embedding_generation_benchmark._endpoint_runtime_env("/raid/praateekm/hf_cache")

    assert runtime_env == {
        "env_vars": {
            "HOME": "/raid/praateekm/hf_cache/curator_endpoint_home",
            "XDG_CACHE_HOME": "/raid/praateekm/hf_cache/curator_endpoint_home/.cache",
            "TORCH_HOME": "/raid/praateekm/hf_cache/curator_endpoint_home/.cache/torch",
            "HF_HOME": "/raid/praateekm/hf_cache",
            "HF_HUB_CACHE": "/raid/praateekm/hf_cache/hub",
            "TRANSFORMERS_CACHE": "/raid/praateekm/hf_cache/hub",
            "UV_CACHE_DIR": "/raid/praateekm/hf_cache/uv_cache",
        }
    }


def test_in_process_vllm_stage_routes_caches_to_writable_model_cache() -> None:
    cache_dir = "/raid/praateekm/hf_cache"
    [stage] = embedding_generation_benchmark._create_embedding_stages(
        model_identifier="google/embeddinggemma-300m",
        model_variation=embedding_generation_benchmark.EmbeddingModelVariation.VLLM_TEXT_PRETOKENIZED,
        model_inference_batch_size=1024,
        model_num_workers=4,
        max_seq_length=2048,
        embedding_pooling="mean_pooling",
        max_chars=1500,
        cache_dir=cache_dir,
    )

    assert stage.pretokenize is True
    assert stage.cache_dir == cache_dir
    assert stage.runtime_env == embedding_generation_benchmark._endpoint_runtime_env(cache_dir)
    assert stage.num_workers() == 4
    assert stage.max_chars == 1500


def test_nightly_yaml_embedding_entries_use_same_input_slice_and_text_cap() -> None:
    config = yaml.safe_load((REPO_ROOT / "benchmarking" / "nightly-benchmark.yaml").read_text())
    entries = {entry["name"]: entry for entry in config["entries"]}

    in_process_raydata = entries["embedding_generation_raydata"]
    ray_serve = entries["embedding_generation_ray_serve_endpoint"]
    dynamo = entries["embedding_generation_dynamo_endpoint"]

    assert "--model-variation=vllm_text_pretokenized" in in_process_raydata["args"]
    assert in_process_raydata["ray"]["num_gpus"] == 4
    assert in_process_raydata["ray"]["num_cpus"] == 64
    assert ray_serve["script"] == "embedding_generation_benchmark.py"
    assert dynamo["script"] == "embedding_generation_benchmark.py"
    assert ray_serve["ray"]["num_gpus"] == dynamo["ray"]["num_gpus"] == 4
    assert ray_serve["ray"]["num_cpus"] == dynamo["ray"]["num_cpus"] == 64

    assert "--model-variation=ray_serve_endpoint" in ray_serve["args"]
    assert "--load-dataset-ratio=0.2" in ray_serve["args"]
    assert "--endpoint-max-concurrent-requests=64" in ray_serve["args"]
    assert "--endpoint-truncate-prompt-tokens=0" in ray_serve["args"]
    assert "--client-num-workers=16" in ray_serve["args"]
    assert "--server-replicas=4" in ray_serve["args"]
    assert "--endpoint-max-chars=1500" in ray_serve["args"]

    assert "--model-variation=dynamo_endpoint" in dynamo["args"]
    assert "--load-dataset-ratio=0.2" in dynamo["args"]
    assert "--endpoint-max-concurrent-requests=64" in dynamo["args"]
    assert "--endpoint-truncate-prompt-tokens=0" in dynamo["args"]
    assert "--client-num-workers=16" in dynamo["args"]
    assert "--server-replicas=4" in dynamo["args"]
    assert "--endpoint-max-chars=1500" in dynamo["args"]

    assert "--load-dataset-ratio=0.2" in in_process_raydata["args"]
    assert "--max-chars=1500" in in_process_raydata["args"]
