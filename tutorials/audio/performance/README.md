# Audio performance report

Set `performance_report_path` on the terminal `ManifestWriterStage` to the JSON destination:

```yaml
performance_report_path: /output/qwen_omni_performance.json

stages:
  - _target_: nemo_curator.stages.audio.common.ManifestWriterStage
    output_path: /output/qwen_omni_output.jsonl
    performance_report_path: ${performance_report_path}
```

## Exact parity-run performance JSON

```json
{
  "executor": "RayDataExecutor",
  "pipeline": {
    "pipeline_description": "Create and execute a pipeline from a YAML file",
    "pipeline_name": "yaml_pipeline",
    "stages": [
      {
        "batch_size": 1,
        "name": "file_partitioning",
        "num_workers": 1,
        "stage_id": "000:file_partitioning",
        "type": "nemo_curator.stages.file_partitioning.FilePartitioningStage"
      },
      {
        "batch_size": 1,
        "name": "manifest_reader_stage",
        "num_workers": 1,
        "stage_id": "001:manifest_reader_stage",
        "type": "nemo_curator.stages.audio.common.ManifestReaderStage"
      },
      {
        "batch_size": 1,
        "name": "ResampleAudio",
        "num_workers": null,
        "stage_id": "002:ResampleAudio",
        "type": "nemo_curator.stages.audio.tagging.resample_audio.ResampleAudioStage"
      },
      {
        "batch_size": 32,
        "name": "ASR_inference",
        "num_workers": null,
        "stage_id": "003:ASR_inference",
        "type": "nemo_curator.stages.audio.inference.asr.stage.ASRStage"
      },
      {
        "batch_size": 1,
        "name": "manifest_writer",
        "num_workers": 1,
        "stage_id": "004:manifest_writer",
        "type": "nemo_curator.stages.audio.common.ManifestWriterStage"
      }
    ]
  },
  "pipeline_name": "yaml_pipeline",
  "record_count": 19,
  "run_id": "4f77004bd7f44d45a7a7662e4c356f64",
  "schema_version": 1,
  "slurm_array": null,
  "wall_time_s": 221.20097979158163,
  "stage_performance": [
    {
      "stage_id": "000:file_partitioning",
      "stage_start_s": 1786046652.2230403,
      "stage_end_s": 1786046652.2237613,
      "invocation_ids": [
        "dab80aa757ae4201a4edbe183e1fab33"
      ],
      "processing_times_s": [
        0.0007188916206359863
      ]
    },
    {
      "stage_id": "001:manifest_reader_stage",
      "stage_start_s": 1786046652.4255745,
      "stage_end_s": 1786046652.426173,
      "invocation_ids": [
        "d25cef22e18f4178867887fe9a1dab7f"
      ],
      "processing_times_s": [
        0.0005961749702692032
      ]
    },
    {
      "stage_id": "002:ResampleAudio",
      "stage_start_s": 1786046652.98385,
      "stage_end_s": 1786046654.9800158,
      "invocation_ids": [
        "8c6b4477cdae4a4993009fdb65e475b0",
        "ed30faff9c7c4a06bc13a1557f072865",
        "841e9144faff4b87b675ec9e493ecbf5",
        "99ee13a2a11f412da0f065875b876ec8",
        "1e1dff96665742839b095dc137737dbb",
        "0801cb0ec48a4f689c735cce4fa565e8",
        "409fcc27870846d796ded40c974b9e31",
        "4c7cb42318c4413f80063eb3957496bc"
      ],
      "processing_times_s": [
        0.14409693144261837,
        0.09774388000369072,
        0.09984212182462215,
        0.12462898902595043,
        0.13993946090340614,
        0.32044319435954094,
        0.41448018327355385,
        0.183643976226449
      ]
    },
    {
      "stage_id": "003:ASR_inference",
      "stage_start_s": 1786046787.9397295,
      "stage_end_s": 1786046794.3312328,
      "invocation_ids": [
        "d104cee5090641499855dd8c68a6e8d8"
      ],
      "processing_times_s": [
        6.391498176380992
      ]
    },
    {
      "stage_id": "004:manifest_writer",
      "stage_start_s": 1786046794.35574,
      "stage_end_s": 1786046794.3759623,
      "invocation_ids": [
        "0229af48cee84296a76559065084c00d",
        "b1303e62883a41af8779dc8ba940da98",
        "c65ebb6dc01b4adb861f67181b68c129",
        "caba5c2cfed9432aa415cd39be7400af",
        "9442766395374078be13c31b790668dc",
        "466064ac0d7c4256b2a3406cf8784847",
        "6c9bef6bd6f64026921d197367ce6efc",
        "b7d7c02f6f07429383c6e6f8414d164e"
      ],
      "processing_times_s": [
        0.0007843915373086929,
        0.00036756135523319244,
        0.00029454007744789124,
        0.00035044923424720764,
        0.0002884604036808014,
        0.0003480222076177597,
        0.0002736244350671768,
        0.00027731992304325104
      ]
    }
  ]
}
```
