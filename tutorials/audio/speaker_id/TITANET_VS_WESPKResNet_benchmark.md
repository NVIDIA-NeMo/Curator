# TitaNet vs WeSpeaker ResNet293_LM — local benchmark summary

This note summarizes **on-machine** speaker-verification numbers comparing **NVIDIA TitaNet-Large** (NeMo / Hugging Face) and **WeSpeaker ResNet293_LM**, using the **VoxCeleb1-O cleaned** trial list and WeSpeaker’s cosine scoring (`wespeaker/bin/score.py`).

## Setup (what was actually run)

| Item | Detail |
|------|--------|
| **Test audio** | VoxCeleb1 test wavs under `/disk_f_nvd/datasets/voxceleb1/test_wav` |
| **Scale** | **4 874** utterances, **40** speakers (subset of full official VoxCeleb1 test — full set is larger) |
| **Trials** | Cleaned Vox1-O: `veri_test2` → `vox1_O_cleaned.kaldi` (**37 611** pairs) |
| **TitaNet** | `nvidia/speakerverification_en_titanet_large`, extraction via `examples/voxceleb/v2/titanet/` |
| **ResNet293_LM** | Local checkpoint `.../Yodas/wespeaker/models/voxceleb_resnet293_LM/`, eval via `examples/voxceleb/v2/eval_resnet293_lm/` |
| **Metrics** | EER (%), minDCF (p_target=0.01, c_miss=1, c_fa=1), **cosine similarity at EER** (sklearn, same rule as WeSpeaker: accept same-speaker if cosine ≥ threshold) |

## Results table

| Model | Cohort mean (ResNet-style) | EER | Cosine @ EER | minDCF |
|--------|---------------------------|-----|--------------|--------|
| **TitaNet** | No | **0.867%** | 0.332730 | 0.136 |
| **TitaNet** | Yes (smoke cohort†) | **0.686%** | 0.291670 | 0.096 |
| **ResNet293_LM** | Yes (smoke cohort†) | **0.468%** | 0.306390 | 0.055 |

† **Smoke cohort**: Vox1 test list duplicated with prefixed utterance IDs (`vox2mean_…`) so the **mean-subtraction code path** matches `examples/voxceleb/v2/local/score.sh`, **without** real VoxCeleb2-dev audio. **Do not treat smoke rows as official benchmark numbers.**

## Short interpretation

1. **Mean subtraction** (same mechanism as the WeSpeaker v2 recipe) **improved** TitaNet EER in this setup (0.867% → 0.686%).
2. With the **same smoke mean pipeline**, **ResNet293_LM scored better** than TitaNet on this subset (lower EER and minDCF).
3. **Absolute cosine thresholds** are not portable across models or preprocessing; they are valid **within** each scored run under the stated mean/no-mean condition.

## Official / fair comparison (recommended)

To align with the **published WeSpeaker VoxCeleb v2** recipe:

1. Point **`VOXCELEB2_WAV_ROOT`** at the real **VoxCeleb2 dev** wav tree (same layout as `prepare_data.sh`).
2. Re-run **both**:
   - `wespeaker/examples/voxceleb/v2/titanet/run_eval.sh` (with real Vox2, **no** `SMOKE_MEAN_FROM_VOX1`)
   - `wespeaker/examples/voxceleb/v2/eval_resnet293_lm/run_eval.sh` (with real Vox2, **no** smoke)
3. Compare only those two runs (same test wavs, same trials, same cohort-mean policy).

## Recipe locations (WeSpeaker repo)

- TitaNet: `examples/voxceleb/v2/titanet/run_eval.sh`
- ResNet293_LM: `examples/voxceleb/v2/eval_resnet293_lm/run_eval.sh`
- Reference scoring script: `wespeaker/bin/score.py` (cohort mean = mean of `embeddings/vox2_dev/xvector.scp`, then cosine similarity on Vox1 test embeddings)

## References

- TitaNet-Large (EN): [Hugging Face — nvidia/speakerverification_en_titanet_large](https://huggingface.co/nvidia/speakerverification_en_titanet_large)
- WeSpeaker ResNet293_LM: [Hugging Face — Wespeaker/wespeaker-voxceleb-resnet293-LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM)

## Curator clustering default (TitaNet)

With `embedding_normalization=center_global` (batch mean before cosine), the default **AHC cosine threshold** in `run_pipeline.py` / `SpeakerClusteringStage` is **0.292**, matching the **TitaNet + mean-subtraction** row (cosine @ EER ≈ 0.291670) from the table above. Use **0.35–0.40** if you need fewer false merges (stricter same-speaker merges).

---

*Document generated from local eval logs; update this file after runs with full VoxCeleb1 test + real Vox2 dev if you need publication-grade numbers.*
