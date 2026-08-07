# Audio performance report

Set `performance_report_path` on the terminal `ManifestWriterStage` to the JSON destination:

```yaml
performance_report_path: /output/qwen_omni_performance.json

stages:
  - _target_: nemo_curator.stages.audio.common.ManifestWriterStage
    output_path: /output/qwen_omni_output.jsonl
    performance_report_path: ${performance_report_path}
```

## Exact 16-row GPU parity-run performance JSON

The following report was captured from commit `ab7081b325dfabc702a3c6642184c20c13633ac4` using 16 input rows, ASR batch size 8, and one NVIDIA GeForce RTX 3080 Ti. The same inputs produced 16 exact transcript matches against Curator main at `a4470c6fe9b20ec98eb0839939c5e89de8aca3e5`.

`record_count` is 18 because the report contains two ASR batch invocations and sixteen manifest-writer invocations. These are invocation-level measurements and are not duplicated across the sixteen output tasks.

```json
{
  "executor": "RayDataExecutor",
  "pipeline": {
    "pipeline_description": "",
    "pipeline_name": "pr2296-local-gpu-parity-16rows-batch8",
    "stages": [
      {
        "batch_size": 8,
        "name": "ASR_inference",
        "num_workers": null,
        "stage_id": "000:ASR_inference",
        "type": "nemo_curator.stages.audio.inference.asr.asr_nemo.InferenceAsrNemoStage"
      },
      {
        "batch_size": 1,
        "name": "manifest_writer",
        "num_workers": 1,
        "stage_id": "001:manifest_writer",
        "type": "nemo_curator.stages.audio.common.ManifestWriterStage"
      }
    ]
  },
  "pipeline_name": "pr2296-local-gpu-parity-16rows-batch8",
  "record_count": 18,
  "run_id": "af7731bd75af4952892019914a169c60",
  "schema_version": 1,
  "slurm_array": null,
  "wall_time_s": 45.5407141353935,
  "stage_performance": [
    {
      "stage_id": "000:ASR_inference",
      "stage_start_s": 1786134574.6084895,
      "stage_end_s": 1786134576.659894,
      "invocation_ids": [
        "04a8330ad8644ce39d67a187ad99dd0f",
        "b7d32ac82aef405d940c5114db342034"
      ],
      "processing_times_s": [
        1.231179116293788,
        0.8097240729257464
      ]
    },
    {
      "stage_id": "001:manifest_writer",
      "stage_start_s": 1786134575.9373822,
      "stage_end_s": 1786134576.6860292,
      "invocation_ids": [
        "ea7ff488581846e5b2aa5358a37a1a4c",
        "8d17962d370a47febb5d0e25819c8c53",
        "881492c1cc044c9baa7f7bddc62ec453",
        "268ecd30bd28454193da1b949190df62",
        "a89be86e2a404fb3a072cfd185089e5b",
        "a4f250eec2e24f7a895f16861151a6e7",
        "ab6b4bda602040cb83e88ec1739de32c",
        "c13734b85bb342af81776638ed9c8133",
        "9924248b4eec4d39b804783fd0644ed0",
        "50a994f37b894982ba0abecb77b900ad",
        "e6cf2423c16d4ba79b3dadcb38be3609",
        "a1bdcfe7311945e5ac2020f16e2d7c23",
        "f3eef18e552045989dd3d81c13c2b6b6",
        "76bc37156bad464ca2854f8a3f777b38",
        "7602260578d34ded8e8bb00be503f03f",
        "906e0b1857d24726a7853b2226983a84"
      ],
      "processing_times_s": [
        0.00014918949455022812,
        0.00012210663408041,
        0.00011932570487260818,
        0.00016196724027395248,
        8.973944932222366e-05,
        8.482858538627625e-05,
        8.516944944858551e-05,
        9.106192737817764e-05,
        0.0001326901838183403,
        0.00012360140681266785,
        0.00013177469372749329,
        0.00013009924441576004,
        0.00012642797082662582,
        0.00013201497495174408,
        0.0001581888645887375,
        0.00013593677431344986
      ]
    }
  ]
}
```
