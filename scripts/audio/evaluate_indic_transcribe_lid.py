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

# ruff: noqa: INP001

"""Evaluate PyTorch and TRT-LLM Indic Transcribe language identification on FLEURS.

Examples:
    python scripts/audio/evaluate_indic_transcribe_lid.py \
        --backend both \
        --dataset-source hf \
        --languages hi,ur,mr,te,ta,gu \
        --samples-per-lang 100 \
        --pytorch-model /home/sdeva/Documents/workspace/downloads/indic_transcribe/indic-transcribe.nemo \
        --trt-engine-dir /home/sdeva/Documents/workspace/downloads/indic_transcribe/cleanbuild/engine_full \
        --output-dir /tmp/indic_transcribe_lid_eval

    python scripts/audio/evaluate_indic_transcribe_lid.py \
        --backend trt \
        --dataset-source local \
        --local-root /home/sdeva/Documents/workspace/downloads/datasets/asr/test_sets \
        --languages hi,ta \
        --samples-per-lang 50
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FLEURS_CONFIG_BY_LANG = {
    "hi": "hi_in",
    "ur": "ur_pk",
    "mr": "mr_in",
    "te": "te_in",
    "ta": "ta_in",
    "gu": "gu_in",
}
DEFAULT_LANGUAGES = "hi,ur,mr,te,ta,gu"
DEFAULT_PYTORCH_MODEL = "/home/sdeva/Documents/workspace/downloads/indic_transcribe/indic-transcribe.nemo"
DEFAULT_TRT_ENGINE = "/home/sdeva/Documents/workspace/downloads/indic_transcribe/cleanbuild/engine_full"
DEFAULT_LOCAL_ROOT = "/home/sdeva/Documents/workspace/downloads/datasets/asr/test_sets"
TRT_TARGET_SR = 16000
TRT_MIN_DURATION_SAMPLES = 400
NON_LANGUAGE_SPECIAL_TOKENS = frozenset({"pnc", "itn"})
LANGUAGE_CODE_MAX_PARTS = 2
LANGUAGE_CODE_LENGTHS = frozenset({2, 3})
LANGUAGE_COUNTRY_CODE_LENGTH = 2


@dataclass
class LidSample:
    sample_id: str
    true_lang: str
    audio_path: str | None
    audio_array: Any | None
    sample_rate: int | None
    duration: float | None
    text: str | None = None


def _parse_languages(value: str) -> list[str]:
    return [lang.strip().lower() for lang in value.split(",") if lang.strip()]


def _strip_lang_token(token: str) -> str:
    token = str(token).strip()
    if token.startswith("<|") and token.endswith("|>"):
        return token[2:-2].lower()
    return token.strip("<|>").strip("|>").lower()


def _language_code_from_special_token(token: str) -> str | None:
    if not token.startswith("<|") or not token.endswith("|>"):
        return None
    code = token[2:-2].lower()
    if code in NON_LANGUAGE_SPECIAL_TOKENS:
        return None
    parts = code.split("-")
    if len(parts) > LANGUAGE_CODE_MAX_PARTS or not all(part.isalpha() for part in parts):
        return None
    if len(parts[0]) not in LANGUAGE_CODE_LENGTHS:
        return None
    if len(parts) == LANGUAGE_CODE_MAX_PARTS and len(parts[1]) != LANGUAGE_COUNTRY_CODE_LENGTH:
        return None
    return code


def _prepare_runtime_cache_dirs() -> None:
    tmpdir = Path(tempfile.gettempdir())
    for env_var, dirname in {
        "NUMBA_CACHE_DIR": str(tmpdir / "numba_cache"),
        "MPLCONFIGDIR": str(tmpdir / "matplotlib"),
    }.items():
        os.environ.setdefault(env_var, dirname)
        Path(os.environ[env_var]).mkdir(parents=True, exist_ok=True)


def _get_hf_token() -> str | None:
    for env_var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(env_var)
        if token:
            return token
    return None


def _batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    batch_size = max(1, int(batch_size))
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _json_default(obj: Any) -> Any:  # noqa: ANN401
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_hf_fleurs(
    *,
    dataset_name: str,
    split: str,
    languages: list[str],
    samples_per_lang: int | None,
) -> list[LidSample]:
    from datasets import Audio, load_dataset

    samples: list[LidSample] = []
    hf_token = _get_hf_token()
    for lang in languages:
        config = FLEURS_CONFIG_BY_LANG.get(lang)
        if config is None:
            msg = f"No FLEURS config mapping for language '{lang}'"
            raise ValueError(msg)
        ds = load_dataset(dataset_name, config, split=split, streaming=True, token=hf_token)
        ds = ds.cast_column("audio", Audio(sampling_rate=None))
        for idx, row in enumerate(ds):
            if samples_per_lang is not None and idx >= samples_per_lang:
                break
            audio = row["audio"]
            samples.append(
                LidSample(
                    sample_id=f"{lang}:{idx}",
                    true_lang=lang,
                    audio_path=audio.get("path"),
                    audio_array=audio["array"],
                    sample_rate=int(audio["sampling_rate"]),
                    duration=None,
                    text=row.get("transcription") or row.get("raw_transcription"),
                )
            )
    return samples


def load_local_fleurs(
    *,
    local_root: Path,
    languages: list[str],
    samples_per_lang: int | None,
) -> list[LidSample]:
    samples: list[LidSample] = []
    for lang in languages:
        manifest = local_root / f"{lang}-IN" / "fleurs.json"
        if not manifest.exists():
            print(f"Skipping {lang}: local manifest not found at {manifest}")
            continue
        rows = _read_jsonl(manifest)
        if samples_per_lang is not None:
            rows = rows[:samples_per_lang]
        for idx, row in enumerate(rows):
            audio_path = Path(row["audio_filepath"])
            if not audio_path.is_absolute():
                audio_path = manifest.parent / audio_path
            samples.append(
                LidSample(
                    sample_id=f"{lang}:{idx}",
                    true_lang=lang,
                    audio_path=str(audio_path),
                    audio_array=None,
                    sample_rate=int(row.get("sample_rate") or row.get("sampling_rate") or 16000),
                    duration=row.get("duration"),
                    text=row.get("text"),
                )
            )
    return samples


def load_audio(sample: LidSample, target_sr: int | None = None) -> tuple[Any, int]:
    import numpy as np

    if sample.audio_array is not None:
        audio = np.asarray(sample.audio_array, dtype=np.float32)
        sr = int(sample.sample_rate or target_sr or 16000)
        if target_sr is not None and sr != target_sr:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return audio.astype(np.float32), sr

    if sample.audio_path is None:
        msg = f"Sample {sample.sample_id} has neither audio array nor audio path"
        raise ValueError(msg)
    import librosa

    audio, sr = librosa.load(sample.audio_path, sr=target_sr, mono=True)
    return audio.astype(np.float32), int(sr)


def lens_to_mask(lens: Any, max_length: int) -> Any:  # noqa: ANN401
    import torch

    batch_size = lens.shape[0]
    arange = torch.arange(max_length, device=lens.device)
    return arange.expand(batch_size, max_length) < lens.unsqueeze(1)


def pytorch_lid(
    *,
    samples: list[LidSample],
    model_path: str,
    batch_size: int,
    device: str,
) -> list[dict[str, Any]]:
    import torch
    from nemo.collections.asr.models import EncDecMultiTaskModel
    from nemo.collections.common.prompts.canary2 import Canary2PromptFormatter

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = EncDecMultiTaskModel.restore_from(model_path, map_location=device)
    model = model.to(device)
    model.eval()

    target_sr = int(getattr(model.preprocessor, "_sample_rate", 16000))
    prompt_formatter = Canary2PromptFormatter(tokenizer=model.tokenizer)
    turns = [
        {
            "role": "user",
            "slots": {
                "decodercontext": "",
                "source_lang": "<|mr|>",
                "target_lang": "<|en|>",
                "emotion": "<|emo:undefined|>",
                "pnc": "<|pnc|>",
                "itn": "<|noitn|>",
                "diarize": "<|nodiarize|>",
                "timestamp": "<|notimestamp|>",
                "prompt_language": "spl_tokens",
                "romanized": "<|noromanized|>",
            },
        }
    ]
    prompt_ids = prompt_formatter.encode_dialog(turns)["input_ids"][:3].to(device)

    predictions: list[dict[str, Any]] = []
    for batch in _batched(samples, batch_size):
        audios = []
        audio_lens = []
        for sample in batch:
            audio, _sr = load_audio(sample, target_sr=target_sr)
            audios.append(torch.tensor(audio, dtype=torch.float32))
            audio_lens.append(len(audio))

        max_len = max(audio_lens)
        audio_tensor = torch.zeros(len(audios), max_len, dtype=torch.float32)
        for i, audio in enumerate(audios):
            audio_tensor[i, : len(audio)] = audio
        audio_tensor = audio_tensor.to(device)
        audio_len_tensor = torch.tensor(audio_lens, device=device)

        with torch.inference_mode():
            processed_signal, processed_signal_len = model.preprocessor(
                input_signal=audio_tensor,
                length=audio_len_tensor,
            )
            encoded, encoded_len = model.encoder(audio_signal=processed_signal, length=processed_signal_len)
            enc_states = model.encoder_decoder_proj(encoded.permute(0, 2, 1))
            enc_mask = lens_to_mask(encoded_len, enc_states.shape[1]).to(enc_states.dtype)
            if getattr(model, "use_transf_encoder", False):
                enc_states = model.transf_encoder(encoder_states=enc_states, encoder_mask=enc_mask)

            decoder_input = prompt_ids.unsqueeze(0).expand(len(batch), -1).contiguous()
            dec_states = model.transf_decoder(
                input_ids=decoder_input,
                decoder_mask=None,
                encoder_embeddings=enc_states,
                encoder_mask=enc_mask,
            )
            logits = model.log_softmax(hidden_states=dec_states[:, -1:])
            probs = torch.exp(logits.squeeze(1))
            topk_probs, topk_indices = probs.topk(5, dim=-1)

        for idx, sample in enumerate(batch):
            top5 = []
            for token_id, prob in zip(topk_indices[idx].tolist(), topk_probs[idx].tolist(), strict=True):
                token = model.tokenizer.ids_to_text([int(token_id)])
                top5.append({"lang": _strip_lang_token(token), "prob": float(prob), "token": token})
            pred_lang = top5[0]["lang"] if top5 else ""
            predictions.append(
                {
                    "sample_id": sample.sample_id,
                    "true_lang": sample.true_lang,
                    "pred_lang": pred_lang,
                    "correct": pred_lang == sample.true_lang,
                    "duration": sample.duration,
                    "audio_path": sample.audio_path,
                    "top5_langs": top5,
                }
            )
    return predictions


def trt_lid(  # noqa: PLR0913
    *,
    samples: list[LidSample],
    engine_dir: str,
    batch_size: int,
    candidate_langs: list[str] | None,
    min_duration_sec: float,
    max_duration_sec: float,
) -> list[dict[str, Any]]:
    import torch

    runtime = _load_trt_runtime()
    model = runtime.CanaryTRTLLM(engine_dir, device="cuda:0")
    language_by_token_id = _collect_trt_language_token_ids(model, candidate_langs)
    prompt_ids = _trt_lid_prompt_ids(model)
    if len(prompt_ids) > int(model.decoder.max_input_len):
        msg = (
            f"Indic Transcribe LangID prompt length {len(prompt_ids)} exceeds decoder max_input_len "
            f"{model.decoder.max_input_len}"
        )
        raise ValueError(msg)

    max_samples = int(max_duration_sec * TRT_TARGET_SR)
    predictions: list[dict[str, Any]] = []
    try:
        max_bs = min(max(1, int(batch_size)), max(1, int(model.max_batch_size)))
        for batch in _batched(samples, max_bs):
            audio_signals = []
            audio_lengths = []
            valid_samples = []
            for sample in batch:
                audio, _sr = load_audio(sample, target_sr=TRT_TARGET_SR)
                if len(audio) / TRT_TARGET_SR < min_duration_sec:
                    predictions.append(
                        {
                            "sample_id": sample.sample_id,
                            "true_lang": sample.true_lang,
                            "pred_lang": "",
                            "correct": False,
                            "confidence": 0.0,
                            "duration": sample.duration,
                            "audio_path": sample.audio_path,
                        }
                    )
                    continue
                audio_tensor = torch.tensor(audio, dtype=torch.float32).reshape(-1)[:max_samples]
                audio_signals.append(audio_tensor)
                audio_lengths.append(int(audio_tensor.shape[0]))
                valid_samples.append(sample)
            if not audio_signals:
                continue

            pad_len = min(max(audio_lengths), max_samples)
            padded = [runtime.pad_or_trim(audio, pad_len) for audio in audio_signals]
            durations = [min(max(length, TRT_MIN_DURATION_SAMPLES), pad_len) for length in audio_lengths]
            decoder_input_ids = torch.tensor(prompt_ids, dtype=torch.int64).repeat(len(padded), 1).to(model.device)
            stream = torch.cuda.current_stream("cuda")
            mel, mel_input_lengths = model.preprocessor.get_feats(padded, durations)
            encoder_output, encoder_output_lengths = model.encoder.infer(mel, mel_input_lengths, stream)
            output_ids = model.decoder.generate(
                decoder_input_ids,
                encoder_output,
                encoder_output_lengths,
                max_new_tokens=1,
                num_beams=1,
            )
            for sample, row in zip(valid_samples, output_ids, strict=True):
                pred_lang, confidence = _parse_trt_language(
                    row[0],
                    prompt_ids,
                    model,
                    language_by_token_id,
                )
                predictions.append(
                    {
                        "sample_id": sample.sample_id,
                        "true_lang": sample.true_lang,
                        "pred_lang": pred_lang,
                        "correct": pred_lang == sample.true_lang,
                        "confidence": confidence,
                        "duration": sample.duration,
                        "audio_path": sample.audio_path,
                    }
                )
    finally:
        del model
        torch.cuda.empty_cache()
    return predictions


def _load_trt_runtime() -> Any:  # noqa: ANN401
    try:
        from nemo_curator.stages.audio.inference import indic_transcribe_trtllm_runtime
    except Exception as exc:
        repo_root = Path(__file__).resolve().parents[2]
        runtime_path = repo_root / "nemo_curator/stages/audio/inference/indic_transcribe_trtllm_runtime.py"
        spec = importlib.util.spec_from_file_location("_indic_transcribe_trtllm_runtime_eval", runtime_path)
        if spec is None or spec.loader is None:
            msg = f"Could not load TRT runtime from {runtime_path}"
            raise ImportError(msg) from exc
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as load_exc:
            msg = f"Could not import TRT runtime from {runtime_path}"
            raise ImportError(msg) from load_exc
        return module
    else:
        return indic_transcribe_trtllm_runtime


def _collect_trt_language_token_ids(model: Any, candidate_langs: list[str] | None) -> dict[int, str]:  # noqa: ANN401
    candidates = frozenset(candidate_langs) if candidate_langs is not None else None
    languages = {}
    for token_id, token in getattr(model.tokenizer, "id_to_token", {}).items():
        code = _language_code_from_special_token(str(token))
        if code is None:
            continue
        if candidates is not None and code not in candidates:
            continue
        languages[int(token_id)] = code
    if not languages:
        msg = "TRT tokenizer has no language special tokens matching candidate languages"
        raise RuntimeError(msg)
    return languages


def _trt_lid_prompt_ids(model: Any) -> list[int]:  # noqa: ANN401
    if model.tokenizer.prompt_format == "canary2":
        return model.tokenizer.encode("<|startofcontext|> <|startoftranscript|> <|emo:undefined|>")
    return model.tokenizer.encode("<|startoftranscript|>")


def _parse_trt_language(
    sequence: list[int],
    prompt_ids: list[int],
    model: Any,  # noqa: ANN401
    language_by_token_id: dict[int, str],
) -> tuple[str, float]:
    sequence = [int(token_id) for token_id in sequence]
    if sequence[: len(prompt_ids)] == prompt_ids:
        sequence = sequence[len(prompt_ids) :]
    eos_id = getattr(model.tokenizer, "eos_id", None)
    pad_id = getattr(model.tokenizer, "pad_id", None)
    for token_id in sequence:
        if token_id == eos_id:
            break
        if token_id == pad_id:
            continue
        lang = language_by_token_id.get(token_id)
        if lang is not None:
            return lang, 1.0
    return "", 0.0


def summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    by_lang: dict[str, dict[str, Any]] = {}
    correct_total = 0
    for pred in predictions:
        lang = pred["true_lang"]
        stats = by_lang.setdefault(lang, {"total": 0, "correct": 0, "accuracy": 0.0})
        stats["total"] += 1
        if pred["correct"]:
            stats["correct"] += 1
            correct_total += 1
    for stats in by_lang.values():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] else 0.0
    total = len(predictions)
    return {
        "total": total,
        "correct": correct_total,
        "accuracy": correct_total / total if total else 0.0,
        "by_lang": dict(sorted(by_lang.items())),
    }


def build_comparison(metrics: dict[str, Any]) -> dict[str, Any]:
    if "pytorch" not in metrics or "trt" not in metrics:
        return {}
    langs = sorted(set(metrics["pytorch"]["by_lang"]) | set(metrics["trt"]["by_lang"]))
    by_lang = {}
    for lang in langs:
        pt = metrics["pytorch"]["by_lang"].get(lang, {})
        trt = metrics["trt"]["by_lang"].get(lang, {})
        by_lang[lang] = {
            "pytorch_accuracy": pt.get("accuracy"),
            "trt_accuracy": trt.get("accuracy"),
            "trt_minus_pytorch": (
                trt.get("accuracy") - pt.get("accuracy")
                if "accuracy" in pt and "accuracy" in trt
                else None
            ),
        }
    return {
        "pytorch_accuracy": metrics["pytorch"]["accuracy"],
        "trt_accuracy": metrics["trt"]["accuracy"],
        "trt_minus_pytorch": metrics["trt"]["accuracy"] - metrics["pytorch"]["accuracy"],
        "by_lang": by_lang,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Indic Transcribe LID on FLEURS.")
    parser.add_argument("--backend", choices=["pytorch", "trt", "both"], default="both")
    parser.add_argument("--dataset-source", choices=["hf", "local"], default="hf")
    parser.add_argument("--dataset-name", default="google/fleurs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--languages", default=DEFAULT_LANGUAGES)
    parser.add_argument("--samples-per-lang", type=int, default=100, help="Use 0 to evaluate all samples.")
    parser.add_argument("--local-root", default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--pytorch-model", default=DEFAULT_PYTORCH_MODEL)
    parser.add_argument("--trt-engine-dir", default=DEFAULT_TRT_ENGINE)
    parser.add_argument("--pytorch-batch-size", type=int, default=4)
    parser.add_argument("--trt-batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', 'cuda:0', or 'cpu' for PyTorch.")
    parser.add_argument("--min-duration-sec", type=float, default=0.0)
    parser.add_argument("--max-duration-sec", type=float, default=30.0)
    parser.add_argument("--output-dir", default="indic_transcribe_lid_eval")
    return parser.parse_args()


def main() -> None:
    _prepare_runtime_cache_dirs()
    args = parse_args()
    languages = _parse_languages(args.languages)
    samples_per_lang = None if args.samples_per_lang == 0 else args.samples_per_lang

    if args.dataset_source == "hf":
        samples = load_hf_fleurs(
            dataset_name=args.dataset_name,
            split=args.split,
            languages=languages,
            samples_per_lang=samples_per_lang,
        )
    else:
        samples = load_local_fleurs(
            local_root=Path(args.local_root),
            languages=languages,
            samples_per_lang=samples_per_lang,
        )
    if not samples:
        msg = "No samples loaded"
        raise RuntimeError(msg)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        output_dir / "manifest.jsonl",
        [
            {
                "sample_id": sample.sample_id,
                "true_lang": sample.true_lang,
                "audio_path": sample.audio_path,
                "sample_rate": sample.sample_rate,
                "duration": sample.duration,
                "text": sample.text,
            }
            for sample in samples
        ],
    )

    metrics: dict[str, Any] = {}
    if args.backend in {"pytorch", "both"}:
        print(f"Running PyTorch LID on {len(samples)} samples")
        preds = pytorch_lid(
            samples=samples,
            model_path=args.pytorch_model,
            batch_size=args.pytorch_batch_size,
            device=args.device,
        )
        _write_jsonl(output_dir / "predictions_pytorch.jsonl", preds)
        metrics["pytorch"] = summarize(preds)

    if args.backend in {"trt", "both"}:
        print(f"Running TRT LID on {len(samples)} samples")
        preds = trt_lid(
            samples=samples,
            engine_dir=args.trt_engine_dir,
            batch_size=args.trt_batch_size,
            candidate_langs=languages,
            min_duration_sec=args.min_duration_sec,
            max_duration_sec=args.max_duration_sec,
        )
        _write_jsonl(output_dir / "predictions_trt.jsonl", preds)
        metrics["trt"] = summarize(preds)

    comparison = build_comparison(metrics)
    result = {
        "dataset_source": args.dataset_source,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "languages": languages,
        "num_samples": len(samples),
        "metrics": metrics,
        "comparison": comparison,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
