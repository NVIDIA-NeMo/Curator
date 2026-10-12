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

"""Generation stage: numeric entities with a canonical written value and a spoken reading.

Every entry pairs a written form (``"4820173"``, ``"-3.045"``, ``"3/5"``, ``"$1234.56"``,
``"472000000"``, ``"12.5%"``) with a deterministic English reading that the scorer's parser
converts back to the same value. The generator is seeded, so a configuration reproduces the same
set on any machine.
"""

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from loguru import logger

from nemo_curator.stages.audio.numpipe.numeric_norm import (
    DEN_WORDS,
    DEN_WORDS_SINGULAR,
    int_to_cardinal,
    int_to_digit_words,
    parse_numeric,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, EmptyTask

ALL_KINDS = ("integer", "decimal", "fraction", "money", "magnitude", "percent")
DEFAULT_KINDS = ("integer", "decimal", "fraction", "money")
MAGNITUDES = (("thousand", 10**3), ("million", 10**6), ("billion", 10**9), ("trillion", 10**12))

_LONG_RANGE_START = 11
_SHORT_RANGE_END = 10
_MAX_REJECTION_DRAWS = 50
_MIN_OVER_DENOMINATOR = 13
_MAX_OVER_DENOMINATOR = 99
_WITH_CENTS_PROB = 0.7
_MONEY_MAGNITUDE_PROB = 0.3
_PERCENT_SMALL_PROB = 0.85
_PERCENT_DECIMAL_PROB = 0.3
_CARDINAL_PROB = 0.5


@dataclass(frozen=True)
class GenerationConfig:
    """Immutable generation configuration.

    Args:
        seed: Random seed; the same configuration always yields the same entries.
        count: Number of entries.
        kinds: Entity kinds to draw from (subset of :data:`ALL_KINDS`).
        kind_weights: Optional per-kind sampling weights, aligned with ``kinds``.
        min_digits: Minimum digits of the integer part (dollars for money).
        max_digits: Maximum digits of the integer part.
        min_frac_places: Minimum decimal places for decimals.
        max_frac_places: Maximum decimal places for decimals.
        neg_frac: Share of negative numbers (fractions are never negative).
        long_frac: Share of entries drawn from the 11+ digit range (length oversampling).
        oh_frac: Share of digit-by-digit readings that read zero as "oh".
        ordinal_frac: Share of fractions in ordinal style ("three fifths") versus "N over M".
        blocklist: Canonical values (or signed digit signatures) that must not be generated.
        allow_fraction_overlap: Let ordinal fractions bypass the blocklist (their value space is
            about sixty-six values, so a value-level hold-out is meaningless there).
    """

    seed: int = 20260903
    count: int = 100
    kinds: tuple[str, ...] = DEFAULT_KINDS
    kind_weights: tuple[float, ...] | None = None
    min_digits: int = 4
    max_digits: int = 20
    min_frac_places: int = 1
    max_frac_places: int = 6
    neg_frac: float = 0.25
    long_frac: float = 0.0
    oh_frac: float = 0.0
    ordinal_frac: float = 0.5
    blocklist: frozenset[str] = field(default_factory=frozenset)
    allow_fraction_overlap: bool = False

    def __post_init__(self) -> None:
        unknown = [k for k in self.kinds if k not in ALL_KINDS]
        if unknown:
            msg = f"unknown kinds {unknown}; choices: {ALL_KINDS}"
            raise ValueError(msg)
        if self.min_digits < 1 or self.max_digits < self.min_digits:
            msg = "require 1 <= min_digits <= max_digits"
            raise ValueError(msg)
        for name in ("neg_frac", "long_frac", "oh_frac", "ordinal_frac"):
            if not 0 <= getattr(self, name) <= 1:
                msg = f"{name} must be in [0, 1]"
                raise ValueError(msg)
        if self.kind_weights is not None and len(self.kind_weights) != len(self.kinds):
            msg = f"kind_weights has {len(self.kind_weights)} values but there are {len(self.kinds)} kinds"
            raise ValueError(msg)


def _rand_digits(rng: random.Random, num_digits: int) -> str:
    lower = 10 ** (num_digits - 1)
    return str(rng.randint(lower, 10**num_digits - 1))


def _sample_len(rng: random.Random, cfg: GenerationConfig) -> int:
    """Digit count, oversampling the long (11+) range with probability ``long_frac``."""
    if cfg.long_frac > 0 and cfg.max_digits >= _LONG_RANGE_START and rng.random() < cfg.long_frac:
        return rng.randint(max(_LONG_RANGE_START, cfg.min_digits), cfg.max_digits)
    upper = min(_SHORT_RANGE_END, cfg.max_digits) if cfg.max_digits >= _SHORT_RANGE_END else cfg.max_digits
    return rng.randint(cfg.min_digits, upper)


def _blocked(number: str, blocklist: frozenset[str]) -> bool:
    if not blocklist:
        return False
    parsed = parse_numeric(number)
    if parsed is None:
        return False
    canon = parsed.canonical()
    sign = "-" if parsed.sign < 0 else ""
    digit_sig = sign + "".join(ch for ch in canon if ch.isdigit())
    return canon in blocklist or digit_sig in blocklist


def _sign(rng: random.Random, cfg: GenerationConfig) -> tuple[str, str]:
    """(canonical prefix, spoken prefix); negative with probability ``neg_frac``."""
    if rng.random() < cfg.neg_frac:
        return "-", rng.choice(["minus", "negative"]) + " "
    return "", ""


def _digit_words(digits: str, oh: bool) -> str:
    """Digit-by-digit words; with ``oh`` every zero is read "oh"."""
    words = int_to_digit_words(digits)
    return words.replace("zero", "oh") if oh else words


def _int_reading(rng: random.Random, digits: str, cfg: GenerationConfig) -> tuple[str, str, bool]:
    """Pick a reading style for an integer part -> (style, words, oh_flag)."""
    if rng.random() < _CARDINAL_PROB:
        oh = rng.random() < cfg.oh_frac
        return "digit_by_digit", _digit_words(digits, oh), oh
    return "cardinal", int_to_cardinal(int(digits)), False


def gen_integer(rng: random.Random, cfg: GenerationConfig) -> dict:
    digits = _rand_digits(rng, _sample_len(rng, cfg))
    can_sign, spoken_sign = _sign(rng, cfg)
    style, words, oh = _int_reading(rng, digits, cfg)
    return {
        "kind": "integer",
        "style": style,
        "oh": oh,
        "number": can_sign + digits,
        "spoken_ref": spoken_sign + words,
    }


def gen_decimal(rng: random.Random, cfg: GenerationConfig) -> dict:
    int_digits = _rand_digits(rng, _sample_len(rng, cfg))
    places = rng.randint(cfg.min_frac_places, cfg.max_frac_places)
    frac = "".join(str(rng.randrange(10)) for _ in range(places))
    can_sign, spoken_sign = _sign(rng, cfg)
    style, int_words, oh = _int_reading(rng, int_digits, cfg)
    spoken = f"{spoken_sign}{int_words} point {_digit_words(frac, oh)}"
    return {
        "kind": "decimal",
        "style": style,
        "oh": oh,
        "number": f"{can_sign}{int_digits}.{frac}",
        "spoken_ref": spoken,
    }


def gen_fraction(rng: random.Random, cfg: GenerationConfig) -> dict:
    if rng.random() < cfg.ordinal_frac:  # ordinal style, small proper fraction: "three fifths"
        den = rng.choice(sorted(DEN_WORDS))
        num = rng.randint(1, den - 1)
        den_word = DEN_WORDS_SINGULAR[den] if num == 1 else DEN_WORDS[den]
        return {
            "kind": "fraction",
            "style": "ordinal",
            "number": f"{num}/{den}",
            "spoken_ref": f"{int_to_cardinal(num)} {den_word}",
        }
    den = rng.randint(_MIN_OVER_DENOMINATOR, _MAX_OVER_DENOMINATOR)
    num = rng.randint(1, den - 1)
    spoken = f"{int_to_cardinal(num)} over {int_to_cardinal(den)}"
    return {"kind": "fraction", "style": "over", "number": f"{num}/{den}", "spoken_ref": spoken}


def gen_money(rng: random.Random, cfg: GenerationConfig) -> dict:
    dollars = _rand_digits(rng, _sample_len(rng, cfg))
    can_sign, spoken_sign = _sign(rng, cfg)
    style, dollar_words, oh = _int_reading(rng, dollars, cfg)
    unit = "dollar" if dollars == "1" else "dollars"
    if rng.random() < _WITH_CENTS_PROB:
        cents = rng.randint(1, 99)
        cent_unit = "cent" if cents == 1 else "cents"
        spoken = f"{spoken_sign}{dollar_words} {unit} and {int_to_cardinal(cents)} {cent_unit}"
        return {
            "kind": "money",
            "style": style,
            "oh": oh,
            "number": f"{can_sign}${dollars}.{cents:02d}",
            "spoken_ref": spoken,
        }
    return {
        "kind": "money",
        "style": style,
        "oh": oh,
        "number": f"{can_sign}${dollars}",
        "spoken_ref": f"{spoken_sign}{dollar_words} {unit}",
    }


def gen_magnitude(rng: random.Random, cfg: GenerationConfig) -> dict:
    """Round magnitudes as people say them: "four hundred seventy two million", "sixty million dollars".

    The canonical form is the full digit string (``472000000``), which the parser treats as equal
    to ``"472 million"``. Random-digit generation almost never produces these, while real-world
    sets are full of them.
    """
    unit, mult = rng.choice(MAGNITUDES)
    n = rng.randint(1, 999)
    value = n * mult
    words = f"{int_to_cardinal(n)} {unit}"
    can_sign, spoken_sign = _sign(rng, cfg)
    if rng.random() < _MONEY_MAGNITUDE_PROB:
        return {
            "kind": "money",
            "style": "magnitude",
            "oh": False,
            "number": f"{can_sign}${value}",
            "spoken_ref": f"{spoken_sign}{words} dollars",
        }
    return {
        "kind": "integer",
        "style": "magnitude",
        "oh": False,
        "number": f"{can_sign}{value}",
        "spoken_ref": spoken_sign + words,
    }


def gen_percent(rng: random.Random, _cfg: GenerationConfig) -> dict:
    """Percentages as spoken in business speech: "fifteen percent", "twelve point five percent".

    The canonical form keeps the unit (``"15%"``), and the parser treats percent as its own kind,
    so ``"15"`` is not accepted for ``"15%"``.
    """
    n = rng.randint(0, 100) if rng.random() < _PERCENT_SMALL_PROB else rng.randint(101, 999)
    if rng.random() < _PERCENT_DECIMAL_PROB:
        tenth = rng.randint(1, 9)
        spoken = f"{int_to_cardinal(n)} point {int_to_cardinal(tenth)} percent"
        return {"kind": "percent", "style": "cardinal", "oh": False, "number": f"{n}.{tenth}%", "spoken_ref": spoken}
    return {
        "kind": "percent",
        "style": "cardinal",
        "oh": False,
        "number": f"{n}%",
        "spoken_ref": f"{int_to_cardinal(n)} percent",
    }


_GENERATORS: dict[str, Callable[[random.Random, GenerationConfig], dict]] = {
    "integer": gen_integer,
    "decimal": gen_decimal,
    "fraction": gen_fraction,
    "money": gen_money,
    "magnitude": gen_magnitude,
    "percent": gen_percent,
}


def _draw(rng: random.Random, cfg: GenerationConfig, index: int) -> tuple[dict, bool]:
    """Rejection-sample one entry against the blocklist -> (row, kept_despite_overlap)."""
    for _ in range(_MAX_REJECTION_DRAWS):
        kind = (
            rng.choices(list(cfg.kinds), weights=list(cfg.kind_weights))[0]
            if cfg.kind_weights
            else cfg.kinds[rng.randrange(len(cfg.kinds))]
        )
        row = _GENERATORS[kind](rng, cfg)
        if not _blocked(row["number"], cfg.blocklist):
            return row, False
        if cfg.allow_fraction_overlap and row["kind"] == "fraction" and row["style"] == "ordinal":
            return row, True
    msg = (
        f"entry {index}: no non-blocked candidate in {_MAX_REJECTION_DRAWS} draws; the blocklist is too dense "
        "for this digit range. Widen the range or allow fraction overlap deliberately."
    )
    raise RuntimeError(msg)


def build_entries(cfg: GenerationConfig) -> list[dict]:
    """Build manifest entries deterministically from ``cfg.seed``.

    Each entry has ``id``, ``seed``, ``kind``, ``style``, ``number`` (canonical written value),
    ``spoken_ref`` (the reading to synthesize), ``num_digits`` and ``negative``.
    """
    rng = random.Random(cfg.seed)  # noqa: S311 - deterministic sampling, not security
    entries: list[dict] = []
    overlap = 0
    for i in range(cfg.count):
        row, kept = _draw(rng, cfg, i)
        overlap += kept
        entries.append(
            {
                "id": f"ent_{i:05d}",
                "seed": cfg.seed,
                **row,
                "num_digits": sum(ch.isdigit() for ch in row["number"]),
                "negative": row["number"].startswith("-"),
            }
        )
    if overlap:
        logger.info(f"{overlap} ordinal fraction(s) kept despite blocklist overlap")
    return entries


def verify_roundtrip(entries: list[dict]) -> float:
    """Share of entries whose ``spoken_ref`` parses back to ``number`` (must be 1.0 for a valid set).

    Rows that read zero as "oh" are parsed with the homophone table, which is exactly what "oh"
    relies on at evaluation time; everything else stays strict so generator bugs cannot hide.
    """
    matches = 0
    for e in entries:
        parsed = parse_numeric(e["spoken_ref"], homophones=bool(e.get("oh")))
        if parsed is not None and parsed == parse_numeric(e["number"]):
            matches += 1
        else:
            logger.warning(f"round-trip mismatch id={e['id']} number={e['number']} spoken={e['spoken_ref']!r}")
    return matches / len(entries) if entries else 0.0


@dataclass
class NumericEntityGeneratorStage(ProcessingStage[EmptyTask, AudioTask]):
    """Source stage: emit one :class:`AudioTask` per generated numeric entity.

    Place it first in a pipeline; it runs once and fans out ``config.count`` tasks. The set is
    verified to round-trip through the parser before anything is emitted.

    Args:
        config: :class:`GenerationConfig` (or a dict with its fields, for YAML configs).
        dataset_name: ``dataset_name`` written on every emitted task.
    """

    config: GenerationConfig | dict = field(default_factory=GenerationConfig)
    dataset_name: str = "numpipe"
    name: str = "numeric_entity_generator"

    def __post_init__(self) -> None:
        if isinstance(self.config, dict):
            cfg = dict(self.config)
            for key in ("kinds", "kind_weights"):
                if isinstance(cfg.get(key), list):
                    cfg[key] = tuple(cfg[key])
            if isinstance(cfg.get("blocklist"), (list, set)):
                cfg["blocklist"] = frozenset(cfg["blocklist"])
            self.config = GenerationConfig(**cfg)

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["id", "kind", "style", "number", "spoken_ref", "num_digits", "negative"]

    def process(self, task: EmptyTask) -> list[AudioTask]:
        t0 = time.perf_counter()
        entries = build_entries(self.config)
        acc = verify_roundtrip(entries)
        if acc < 1.0:
            msg = f"round-trip verification failed ({acc:.1%}); reference or parser bug, fix before synthesis"
            raise RuntimeError(msg)
        tasks = [
            AudioTask(
                task_id=e["id"],
                dataset_name=self.dataset_name,
                data=e,
                _metadata=task._metadata,
                _stage_perf=list(task._stage_perf),
            )
            for e in entries
        ]
        self._log_metrics({"process_time": time.perf_counter() - t0, "entries_generated": len(tasks)})
        logger.info(f"generated {len(tasks)} numeric entities (seed {self.config.seed}), round-trip 100%")
        return tasks

    def num_workers(self) -> int | None:
        return 1
