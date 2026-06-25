# NeMo Data Designer Synthetic Data Generation

This tutorial shows how to use NeMo Data Designer with NeMo Curator to generate synthetic medical notes from seed symptom and diagnosis data. It downloads a small CSV dataset, converts it into JSONL seed records, builds a Data Designer configuration with prompt templates and samplers, and runs the generation workflow through a Curator pipeline.

The tutorial supports both local and remote inference. By default, it starts a local Ray Serve + vLLM `InferenceServer` for `openai/gpt-oss-20b`; users can also set a remote provider such as NVIDIA NIM and point Data Designer at that endpoint instead.

Use `ndd_data_generation_example.ipynb` for the notebook walkthrough or `ndd_data_generation_example.py` for the script version of the same workflow.

## Local InferenceServer Notes

When using the local server, the tutorial detects the number of Ray-visible GPUs and uses that value as vLLM `tensor_parallel_size`.

On some PCIe-only multi-GPU systems, NCCL peer-to-peer initialization can hang during vLLM startup. This is most likely to appear when `tensor_parallel_size > 1`.

If the tutorial hangs while starting the local inference server, try one of the following:

```bash
export NCCL_P2P_DISABLE=1
```

or edit the inference-server cell to use single-GPU tensor parallelism:

```python
tensor_parallel_size = 1
```

Then pass that value into `engine_kwargs`:

```python
engine_kwargs={
    "tensor_parallel_size": tensor_parallel_size,
}
```

`NCCL_P2P_DISABLE=1` allows multi-GPU serving to continue but may reduce communication performance. Setting `tensor_parallel_size=1` avoids cross-GPU NCCL collectives for vLLM while leaving the remaining GPUs visible to Ray and the tutorial. Restart Ray or the tutorial kernel after changing NCCL environment variables.
