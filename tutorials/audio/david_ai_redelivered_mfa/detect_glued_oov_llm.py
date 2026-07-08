#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Detect glued OOV tokens with an LLM and write high-confidence repairs to TSV."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

from david_ai_common import PipelineError, run_main
from david_ai_glued_words import try_unglue_word

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

_APOSTROPHE_JUNK_RE = re.compile(r"^'+|^'|'$")
_NON_WORD_RE = re.compile(r"[^a-z0-9'-]+")
_COMMON_GLUE_SUFFIXES = (
    "and",
    "the",
    "you",
    "your",
    "that",
    "this",
    "with",
    "for",
    "but",
    "then",
    "yeah",
    "okay",
    "right",
    "thanks",
    "please",
    "because",
    "really",
    "just",
    "like",
    "well",
    "so",
    "now",
    "here",
    "there",
)


def load_dictionary_words(path: Path) -> set[str]:
    words: set[str] = set()
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            token = line.split("\t", 1)[0].strip().lower()
            if token:
                words.add(token)
    return words


def load_oov_words(path: Path) -> list[str]:
    words: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        token = line.strip()
        if token:
            words.append(token)
    return words


def is_junk_oov(token: str) -> bool:
    if not token or token in {"spn", "sil", "sp"}:
        return True
    if _APOSTROPHE_JUNK_RE.search(token):
        return True
    if _NON_WORD_RE.search(token):
        return True
    if len(token) < 4:
        return True
    return False


def boundary_split_candidate(token: str, dictionary: set[str]) -> str | None:
    lower = token.casefold()
    for i in range(3, len(lower) - 2):
        left, right = lower[:i], lower[i:]
        if left in dictionary and right in dictionary:
            return f"{left} {right}"
    return None


def suffix_glue_candidate(token: str, dictionary: set[str]) -> str | None:
    lower = token.casefold()
    for suffix in _COMMON_GLUE_SUFFIXES:
        if lower.endswith(suffix) and len(lower) > len(suffix) + 2:
            prefix = lower[: -len(suffix)]
            if prefix in dictionary:
                return f"{prefix} {suffix}"
    return None


def prefilter_candidates(
    oov_words: list[str],
    dictionary: set[str],
    *,
    mode: str,
) -> list[tuple[str, str | None]]:
    """Return (word, heuristic_replacement) pairs worth sending to the LLM."""
    out: list[tuple[str, str | None]] = []
    seen: set[str] = set()

    for token in oov_words:
        if token in seen or is_junk_oov(token):
            continue
        if token.casefold() in dictionary:
            continue

        heuristic: str | None = None
        unglue = try_unglue_word(token, dictionary)
        boundary = boundary_split_candidate(token, dictionary)
        suffix = suffix_glue_candidate(token, dictionary)

        include = False
        if mode == "all_alpha_ge6":
            include = len(token) >= 6 and token.replace("'", "").replace("-", "").isalpha()
        elif mode == "boundary":
            include = boundary is not None
            heuristic = boundary
        elif mode == "unglue_only":
            include = unglue is not None
            heuristic = " ".join(unglue) if unglue else None
        else:  # unglue_and_boundary (default)
            include = unglue is not None or boundary is not None or suffix is not None
            if unglue:
                heuristic = " ".join(unglue)
            elif boundary:
                heuristic = boundary
            elif suffix:
                heuristic = suffix

        if not include:
            continue
        seen.add(token)
        out.append((token, heuristic))

    return out


def load_processed_words(checkpoint: Path) -> set[str]:
    if not checkpoint.is_file():
        return set()
    done: set[str] = set()
    for line in checkpoint.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        word = row.get("word")
        if isinstance(word, str):
            done.add(word)
    return done


def append_checkpoint(checkpoint: Path, rows: list[dict]) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt(batch: list[tuple[str, str | None]]) -> str:
    lines = [
        "You classify OOV tokens from MFA-normalized English speech transcripts.",
        "A GLUED token is two or more words accidentally concatenated during normalization,",
        "for example: abandonedand -> abandoned and, abbythanks -> abby thanks.",
        "NOT glued: real rare words (aardvark), names, technical terms, or valid contractions.",
        "Replacements must be lowercase, space-separated words suitable for MFA alignment.",
        "Keep apostrophes inside contractions when appropriate (don't, we're).",
        "Return ONLY valid JSON:",
        '{"results":[{"word":"...","glued":true,"replacement":"...","confidence":0.97,"reason":"..."}]}',
        "confidence is 0-1. Only set glued=true when you are >=0.95 confident.",
        "",
        "Tokens (word | heuristic_split_if_any):",
    ]
    for word, heuristic in batch:
        if heuristic:
            lines.append(f"- {word} | {heuristic}")
        else:
            lines.append(f"- {word} |")
    return "\n".join(lines)


def parse_llm_json(text: str) -> list[dict]:
    text = text.strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return []
        payload = json.loads(text[start : end + 1])
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            return results
    return []


def list_models(*, api_key: str, base_url: str | None, timeout_s: float) -> list[str]:
    """Return model ids advertised by the gateway (connectivity check)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=timeout_s)
    return sorted(m.id for m in client.models.list().data)


def call_llm(
    prompt: str,
    *,
    model: str,
    api_key: str,
    base_url: str | None,
    timeout_s: float,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url or None)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a careful transcript normalization assistant. Respond with JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        timeout=timeout_s,
    )
    content = response.choices[0].message.content
    return content or ""


def write_suggestions_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "word\treplacement\tconfidence\tglued\treason\theuristic\n"
    lines = [header]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row.get("word", "")),
                    str(row.get("replacement", "")),
                    f"{float(row.get('confidence', 0.0)):.4f}",
                    str(bool(row.get("glued", False))),
                    str(row.get("reason", "")).replace("\t", " ").replace("\n", " "),
                    str(row.get("heuristic", "")).replace("\t", " "),
                ]
            )
            + "\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def write_repairs_tsv(path: Path, rows: list[dict], *, min_confidence: float) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    repairs: dict[str, str] = {}
    for row in rows:
        if not row.get("glued"):
            continue
        try:
            confidence = float(row.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if confidence < min_confidence:
            continue
        word = str(row.get("word", "")).strip()
        replacement = str(row.get("replacement", "")).strip()
        if word and replacement and replacement != word:
            repairs[word] = replacement
    lines = [f"{word}\t{replacement}" for word, replacement in sorted(repairs.items())]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(repairs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--oov-list",
        type=Path,
        default=Path("workdir/lexicon/oov_words.txt"),
        help="OOV word list (one token per line)",
    )
    ap.add_argument(
        "--dict",
        type=Path,
        default=Path("workdir/lexicon/english_mfa_davidai_eng.dict"),
        help="MFA dictionary for heuristic prefilter",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("workdir/lexicon/llm_unglue_suggestions.tsv"),
        help="Full LLM suggestion report (all processed tokens)",
    )
    ap.add_argument(
        "--repairs-out",
        type=Path,
        default=Path("workdir/lexicon/unglue_repairs_llm.tsv"),
        help="High-confidence repairs compatible with unglue_repairs.tsv",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("workdir/lexicon/llm_unglue_checkpoint.jsonl"),
        help="Resume checkpoint (one JSON object per processed word)",
    )
    ap.add_argument(
        "--prefilter",
        choices=("unglue_and_boundary", "unglue_only", "boundary", "all_alpha_ge6"),
        default="unglue_and_boundary",
    )
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--max-candidates", type=int, default=0, help="Limit candidates (0 = all)")
    ap.add_argument("--min-confidence", type=float, default=0.95)
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--sleep-s", type=float, default=0.5, help="Pause between LLM batches")
    ap.add_argument("--dry-run", action="store_true", help="Write candidate list only; no LLM calls")
    ap.add_argument(
        "--list-models",
        action="store_true",
        help="List model ids from the gateway and exit (connectivity check).",
    )
    args = ap.parse_args()

    if args.list_models:
        if not args.api_key:
            raise PipelineError("Set OPENAI_API_KEY or pass --api-key to list models")
        logger.info("Querying %s for available models", args.base_url or "default OpenAI endpoint")
        for model_id in list_models(
            api_key=args.api_key,
            base_url=args.base_url or None,
            timeout_s=args.timeout_s,
        ):
            print(model_id)
        return 0

    oov_path = args.oov_list.resolve()
    dict_path = args.dict.resolve()
    if not oov_path.is_file():
        raise PipelineError(f"Missing OOV list: {oov_path}")
    if not dict_path.is_file():
        raise PipelineError(f"Missing dictionary: {dict_path}")

    logger.info("Loading dictionary from %s", dict_path)
    dictionary = load_dictionary_words(dict_path)
    logger.info("Dictionary words: %d", len(dictionary))

    oov_words = load_oov_words(oov_path)
    logger.info("OOV tokens: %d", len(oov_words))

    candidates = prefilter_candidates(oov_words, dictionary, mode=args.prefilter)
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]
    logger.info("LLM candidates after prefilter (%s): %d", args.prefilter, len(candidates))

    if args.dry_run:
        dry_path = args.out.with_suffix(".candidates.txt")
        dry_path.parent.mkdir(parents=True, exist_ok=True)
        dry_path.write_text(
            "\n".join(f"{word}\t{heuristic or ''}" for word, heuristic in candidates) + "\n",
            encoding="utf-8",
        )
        logger.info("Dry run wrote %d candidates to %s", len(candidates), dry_path)
        return 0

    if not args.api_key:
        raise PipelineError("Set OPENAI_API_KEY or pass --api-key for LLM detection")

    processed = load_processed_words(args.checkpoint)
    pending = [(w, h) for w, h in candidates if w not in processed]
    logger.info("Pending LLM batches: %d words (%d already checkpointed)", len(pending), len(processed))

    all_rows: list[dict] = []
    if args.checkpoint.is_file():
        for line in args.checkpoint.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                all_rows.append(json.loads(line))

    batch_size = max(1, args.batch_size)
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        prompt = build_prompt(batch)
        logger.info(
            "LLM batch %d-%d / %d",
            start + 1,
            start + len(batch),
            len(pending),
        )
        try:
            raw = call_llm(
                prompt,
                model=args.model,
                api_key=args.api_key,
                base_url=args.base_url or None,
                timeout_s=args.timeout_s,
            )
        except Exception as exc:
            logger.error("LLM call failed at batch starting %d: %s", start, exc)
            raise PipelineError(f"LLM call failed: {exc}") from exc

        parsed = parse_llm_json(raw)
        by_word = {str(row.get("word", "")): row for row in parsed if row.get("word")}

        batch_rows: list[dict] = []
        for word, heuristic in batch:
            row = by_word.get(word, {})
            batch_rows.append(
                {
                    "word": word,
                    "heuristic": heuristic or "",
                    "glued": bool(row.get("glued", False)),
                    "replacement": str(row.get("replacement", "")).strip(),
                    "confidence": float(row.get("confidence", 0.0) or 0.0),
                    "reason": str(row.get("reason", "")).strip(),
                }
            )
        append_checkpoint(args.checkpoint, batch_rows)
        all_rows.extend(batch_rows)
        if args.sleep_s > 0:
            time.sleep(args.sleep_s)

    write_suggestions_tsv(args.out, all_rows)
    repair_count = write_repairs_tsv(args.repairs_out, all_rows, min_confidence=args.min_confidence)
    logger.info(
        "Wrote %d suggestions to %s; %d repairs (>=%.2f) to %s",
        len(all_rows),
        args.out,
        repair_count,
        args.min_confidence,
        args.repairs_out,
    )
    return 0


if __name__ == "__main__":
    run_main(main)
