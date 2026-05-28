#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Precompute an IPA phoneme vocabulary for AcousticDistractorStage.

Reads a user-provided vocabulary file (one word/phrase per JSON array
entry, or one entry per line if the input is plain text), runs each
entry through ``phonemizer`` (espeak-ng backend) for the requested
language, and writes a JSON mapping ``{word: [phoneme_tokens]}`` for
runtime lookup.

The output of this script is consumed by
``nemo_curator/stages/audio/text_filtering/acoustic_distractor.py``.

Usage
-----
    python scripts/build_phoneme_vocab.py \\
        --vocab vocab.json \\
        --language en-us \\
        --output phoneme_vocab_en.json

Run once per target language; the resulting JSON is small (a few MB
for a 50K word vocab) and is loaded into memory at stage setup().

Dependencies
------------
- ``phonemizer`` (``pip install phonemizer``)
- ``espeak-ng`` system package (``apt install espeak-ng``)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_vocab(vocab_path: Path) -> list[str]:
    """Load vocab from JSON (list of strings) or plain text (one per line)."""
    raw = vocab_path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if isinstance(parsed, list):
        return [str(w).strip() for w in parsed if str(w).strip()]
    if isinstance(parsed, dict):
        return [str(w).strip() for w in parsed if str(w).strip()]
    msg = f"Unsupported vocab JSON structure: top-level {type(parsed).__name__}; expected list or object."
    raise ValueError(msg)


def _dedupe_preserving_order(words: list[str], *, lowercase: bool) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        key = w.lower() if lowercase else w
        if key in seen:
            continue
        seen.add(key)
        out.append(key if lowercase else w)
    return out


def _phonemize_batch(
    words: list[str],
    language: str,
    *,
    batch_size: int,
    n_jobs: int,
) -> list[list[str]]:
    """Run phonemizer in batched mode and return per-word phoneme token lists."""
    from phonemizer import phonemize
    from phonemizer.separator import Separator

    sep = Separator(phone=" ", word=" | ", syllable="")
    results: list[list[str]] = []
    total = len(words)
    for start in range(0, total, batch_size):
        chunk = words[start : start + batch_size]
        ipa_lines = phonemize(
            chunk,
            backend="espeak",
            language=language,
            separator=sep,
            strip=True,
            preserve_punctuation=False,
            with_stress=False,
            njobs=n_jobs,
        )
        for line in ipa_lines:
            tokens = [tok for tok in line.split() if tok and tok != "|"]
            results.append(tokens)
        done = min(start + batch_size, total)
        print(f"[phonemize] {done}/{total}", file=sys.stderr, flush=True)
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build a precomputed IPA phoneme vocabulary for AcousticDistractorStage.")
    ap.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="Path to the input vocab file (JSON array/object of strings, or plain text with one entry per line).",
    )
    ap.add_argument(
        "--language",
        type=str,
        required=True,
        help="espeak-ng language code (e.g. en-us, fr, de, cmn, ja).",
    )
    ap.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the {word: [phoneme_tokens]} JSON output.",
    )
    ap.add_argument(
        "--lowercase",
        action="store_true",
        default=True,
        help="Lowercase vocab entries before phonemization and dedup (default: True).",
    )
    ap.add_argument(
        "--no-lowercase",
        action="store_false",
        dest="lowercase",
        help="Preserve original case (useful for languages without case, e.g. Chinese, Japanese, Thai).",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Number of words passed to phonemizer per call. Larger values are faster but use more memory.",
    )
    ap.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Worker processes for phonemizer. Higher = faster on multi-core, but uses more memory.",
    )
    return ap


def main() -> None:
    args = _build_arg_parser().parse_args()

    vocab_path = Path(args.vocab)
    output_path = Path(args.output)

    if not vocab_path.exists():
        msg = f"Vocab file not found: {vocab_path}"
        raise FileNotFoundError(msg)

    raw_words = _load_vocab(vocab_path)
    if not raw_words:
        msg = f"No vocabulary entries found in {vocab_path}."
        raise ValueError(msg)

    words = _dedupe_preserving_order(raw_words, lowercase=args.lowercase)
    print(
        f"[vocab] loaded {len(raw_words)} entries, {len(words)} unique after dedup "
        f"(lowercase={args.lowercase})",
        file=sys.stderr,
        flush=True,
    )

    phoneme_lists = _phonemize_batch(
        words,
        language=args.language,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
    )

    vocab_out: dict[str, list[str]] = {}
    n_empty = 0
    for word, phonemes in zip(words, phoneme_lists, strict=True):
        if not phonemes:
            n_empty += 1
            continue
        vocab_out[word] = phonemes

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(vocab_out, fh, ensure_ascii=False)

    print(
        f"[vocab] wrote {len(vocab_out)} entries to {output_path} "
        f"(skipped {n_empty} with empty phonemization)",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
