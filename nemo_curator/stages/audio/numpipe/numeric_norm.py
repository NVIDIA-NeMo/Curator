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

"""Value-level parsing of spoken and written numbers.

The parser is the metric's source of truth. It reads integers (digit-by-digit or cardinal),
decimals, fractions, money and percentages, with an optional sign, in words
(``"minus three point one four"``), digits (``"-3.14"``) or the hybrid form ASR models often
emit for long numbers (``"155 trillion 871 billion 544 048"``). It is pure Python.

API:

- :func:`parse_numeric` — text -> :class:`NumValue` (or ``None`` when the text is not a number).
- :func:`numeric_eq` — value equality between a hypothesis and a canonical reference.
- :func:`int_to_cardinal` / :func:`int_to_digit_words` — spoken forms (exact inverses of the parser).
- :func:`perturb_canonical` — one-digit hard negative for judge audits.
"""

import random
import re
from dataclasses import dataclass

from nemo_curator.stages.audio.numpipe.text_utils import apply_homophones

UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
TEENS = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
SCALES = {
    "hundred": 100,
    "thousand": 10**3,
    "million": 10**6,
    "billion": 10**9,
    "trillion": 10**12,
    "quadrillion": 10**15,
    "quintillion": 10**18,
    "sextillion": 10**21,
    "septillion": 10**24,
    "octillion": 10**27,
    "nonillion": 10**30,
    "decillion": 10**33,
}
_SCALE_ORDER = [
    "thousand",
    "million",
    "billion",
    "trillion",
    "quadrillion",
    "quintillion",
    "sextillion",
    "septillion",
    "octillion",
    "nonillion",
    "decillion",
]

DIGIT_WORDS = {v: k for k, v in UNITS.items()}

# Fraction denominators spoken as ordinal plurals (2..12); larger denominators use "N over M".
DEN_WORDS = {
    2: "halves",
    3: "thirds",
    4: "quarters",
    5: "fifths",
    6: "sixths",
    7: "sevenths",
    8: "eighths",
    9: "ninths",
    10: "tenths",
    11: "elevenths",
    12: "twelfths",
}
DEN_WORDS_SINGULAR = {
    2: "half",
    3: "third",
    4: "quarter",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
}
_ORDINAL_TO_DEN = {w: n for n, w in DEN_WORDS.items()} | {w: n for n, w in DEN_WORDS_SINGULAR.items()}
_ORDINAL_TO_DEN |= {"fourth": 4, "fourths": 4}  # accept both "quarters" and "fourths"

_SIGN_WORDS = {"minus", "negative"}
_FILLER_WORDS = {"and", "a", "an", "the"}
_MONEY_WORDS = {"dollars", "dollar", "cents", "cent"}
_MAX_CENTS = 99
_CENTS_DIGITS = 2
_THREE_DIGIT_GROUP = 1000


@dataclass(frozen=True)
class NumValue:
    """Canonical parsed value.

    ``digits`` and ``frac`` are kept as exact digit strings so that spoken zeros stay significant
    (``"three point five zero"`` is not ``"three point five"``).
    """

    kind: str  # integer | decimal | fraction | money | percent
    sign: int = 1
    digits: str = ""  # integer part / numerator / dollars
    frac: str | None = None  # decimal digits / denominator / cents (two digits)

    def canonical(self) -> str:
        """Written form: ``"-3.045"``, ``"3/5"``, ``"$1234.56"``, ``"12.5%"``, ``"4820173"``."""
        sign = "-" if self.sign < 0 else ""
        if self.kind == "decimal":
            return f"{sign}{self.digits}.{self.frac}"
        if self.kind == "fraction":
            return f"{sign}{self.digits}/{self.frac}"
        if self.kind == "money":
            cents = f".{self.frac}" if self.frac is not None else ""
            return f"{sign}${self.digits}{cents}"
        if self.kind == "percent":
            frac = f".{self.frac}" if self.frac is not None else ""
            return f"{sign}{self.digits}{frac}%"
        return f"{sign}{self.digits}"


def values_equal(a: NumValue | None, b: NumValue | None) -> bool:
    """Value equality. Money without spoken cents equals an explicit ``.00``."""
    if a is None or b is None or a.kind != b.kind or a.sign != b.sign:
        return False
    if a.kind == "money":
        return a.digits == b.digits and (a.frac or "00") == (b.frac or "00")
    return a.digits == b.digits and a.frac == b.frac


# ---------------------------------------------------------------------------
# words -> int
# ---------------------------------------------------------------------------


def cardinal_to_int(tokens: list[str]) -> int | None:
    """Scale-parse cardinal tokens to an int.

    Handles the hybrid word/digit form (``"155 trillion 871 billion 91 million 544 048"``): trailing
    digit tokens concatenate as three-digit groups. Returns ``None`` on any unknown token.
    """
    if not tokens:
        return None
    total, current = 0, 0
    prev_was_digits = False
    for tok in tokens:
        if tok.isdigit():
            current = current * _THREE_DIGIT_GROUP + int(tok) if prev_was_digits else current + int(tok)
            prev_was_digits = True
            continue
        prev_was_digits = False
        if tok in UNITS:
            current += UNITS[tok]
        elif tok in TEENS:
            current += TEENS[tok]
        elif tok in TENS:
            current += TENS[tok]
        elif tok == "hundred":
            current = (current or 1) * 100
        elif tok in SCALES:
            total += (current or 1) * SCALES[tok]
            current = 0
        else:
            return None
    return total + current


def _single_digits(tokens: list[str]) -> str | None:
    """Map tokens that are all single digits (words or characters) to a digit string."""
    out: list[str] = []
    for tok in tokens:
        if tok in UNITS:
            out.append(str(UNITS[tok]))
        elif len(tok) == 1 and tok.isdigit():
            out.append(tok)
        else:
            return None
    return "".join(out)


def to_int_digits(tokens: list[str]) -> str | None:
    """Digit-by-digit readings keep leading zeros (``"zero five"`` -> ``"05"``); otherwise scale-parse."""
    if not tokens:
        return None
    single = _single_digits(tokens)
    if single is not None:
        return single
    value = cardinal_to_int(tokens)
    return str(value) if value is not None else None


def _frac_digits(tokens: list[str]) -> str | None:
    """Digits after "point": single digit words or characters, or literal digit runs (``"45"``)."""
    out: list[str] = []
    for tok in tokens:
        if tok in UNITS:
            out.append(str(UNITS[tok]))
        elif tok.isdigit():
            out.append(tok)
        else:
            return None
    return "".join(out) if out else None


# ---------------------------------------------------------------------------
# int -> words (generation; exact inverses of the parsers above)
# ---------------------------------------------------------------------------


def int_to_digit_words(digits: str) -> str:
    """Digit-by-digit spoken form: ``"405"`` -> ``"four zero five"``."""
    return " ".join(DIGIT_WORDS[int(ch)] for ch in digits)


def _three_digit_words(n: int) -> str:
    parts: list[str] = []
    if n >= 100:  # noqa: PLR2004
        parts.extend([DIGIT_WORDS[n // 100], "hundred"])
        n %= 100
    if n >= 20:  # noqa: PLR2004
        parts.append({v: k for k, v in TENS.items()}[n - n % 10])
        n %= 10
    elif n >= 10:  # noqa: PLR2004
        parts.append({v: k for k, v in TEENS.items()}[n])
        n = 0
    if n:
        parts.append(DIGIT_WORDS[n])
    return " ".join(parts)


def int_to_cardinal(n: int) -> str:
    """Full cardinal reading: ``4820173`` -> ``"four million eight hundred twenty thousand ..."``."""
    if n == 0:
        return "zero"
    groups: list[int] = []
    while n:
        groups.append(n % _THREE_DIGIT_GROUP)
        n //= _THREE_DIGIT_GROUP
    if len(groups) - 2 >= len(_SCALE_ORDER):
        msg = f"number too large for a cardinal reading ({len(groups) * 3} digits)"
        raise ValueError(msg)
    parts: list[str] = []
    for idx in range(len(groups) - 1, -1, -1):
        if not groups[idx]:
            continue
        parts.append(_three_digit_words(groups[idx]))
        if idx:
            parts.append(_SCALE_ORDER[idx - 1])
    return " ".join(parts)


# ---------------------------------------------------------------------------
# text -> NumValue
# ---------------------------------------------------------------------------

_LITERAL_RE = re.compile(
    r"^(?P<sign>minus |negative |-)?\s*(?P<dollar>\$)?\s*"
    r"(?P<int>\d[\d,]*)(?:\.(?P<frac>\d+))?(?:\s*/\s*(?P<den>\d+))?(?P<pct>\s*%)?$"
)


def _parse_literal(s: str) -> NumValue | None:  # noqa: PLR0911
    m = _LITERAL_RE.match(s)
    if not m:
        return None
    sign = -1 if m.group("sign") else 1
    digits = m.group("int").replace(",", "")
    if m.group("pct"):
        if m.group("dollar") or m.group("den"):
            return None
        return NumValue("percent", sign, digits, m.group("frac"))
    if m.group("den"):
        if m.group("dollar") or m.group("frac"):
            return None
        return NumValue("fraction", sign, digits, m.group("den"))
    if m.group("dollar"):
        cents = m.group("frac")
        if cents is not None and len(cents) < _CENTS_DIGITS:
            cents = (cents + "0")[:_CENTS_DIGITS]  # "$3.5" means $3.50, not $3.05
        return NumValue("money", sign, digits, cents)
    if m.group("frac") is not None:
        return NumValue("decimal", sign, digits, m.group("frac"))
    return NumValue("integer", sign, digits)


def _tokenize(text: str, homophones: bool) -> tuple[int, list[str]]:
    s = text.lower().replace("-", " ").replace(",", " ")
    tokens = re.findall(r"[a-z]+|\d+|\$", s)
    if homophones:
        tokens = apply_homophones(" ".join(tokens)).split()
    tokens = [t for t in tokens if t not in _FILLER_WORDS]
    sign = 1
    while tokens and tokens[0] in _SIGN_WORDS:
        sign = -sign
        tokens = tokens[1:]
    return sign, tokens


def _parse_money(sign: int, tokens: list[str]) -> NumValue | None:
    tokens = [t for t in tokens if t != "$"]
    if "dollars" in tokens or "dollar" in tokens:
        cut = tokens.index("dollars" if "dollars" in tokens else "dollar")
        dollars = to_int_digits(tokens[:cut])
        rest = [t for t in tokens[cut + 1 :] if t not in ("cents", "cent")]
        if dollars is None:
            return None
        if not rest:
            return NumValue("money", sign, dollars)
        cents = to_int_digits(rest)
        if cents is None or int(cents) > _MAX_CENTS:
            return None
        return NumValue("money", sign, dollars, f"{int(cents):02d}")
    rest = [t for t in tokens if t not in ("cents", "cent")]  # cents only: "seventy five cents"
    cents = to_int_digits(rest)
    if cents is None or int(cents) > _MAX_CENTS:
        return None
    return NumValue("money", sign, "0", f"{int(cents):02d}")


def _parse_fraction(sign: int, tokens: list[str]) -> NumValue | None:
    if "over" in tokens:
        cut = tokens.index("over")
        num = to_int_digits(tokens[:cut])
        den = to_int_digits(tokens[cut + 1 :])
        if num is None or den is None:
            return None
        return NumValue("fraction", sign, num, den)
    den_val = _ORDINAL_TO_DEN.get(tokens[-1])
    num = to_int_digits(tokens[:-1])
    if den_val is None or num is None:
        return None
    return NumValue("fraction", sign, num, str(den_val))


def _parse_percent(sign: int, body: list[str]) -> NumValue | None:
    if not body or "$" in body or any(t in _MONEY_WORDS for t in body):
        return None
    if "point" in body:
        cut = body.index("point")
        int_digits = to_int_digits(body[:cut]) if cut else "0"
        frac = _frac_digits(body[cut + 1 :])
        if int_digits is None or frac is None:
            return None
        return NumValue("percent", sign, int_digits, frac)
    int_digits = to_int_digits(body)
    return NumValue("percent", sign, int_digits) if int_digits is not None else None


def parse_numeric(text: str, *, homophones: bool = False) -> NumValue | None:  # noqa: PLR0911
    """Parse any supported number rendering (words, digits or hybrid) to a :class:`NumValue`.

    Args:
        text: The hypothesis or reference text.
        homophones: Map digit-word homophones first (``"for two oh"`` -> ``"four two zero"``).
    """
    if not text:
        return None
    s = re.sub(r"\s+", " ", text.strip().rstrip(".")).strip()
    literal = _parse_literal(s.lower())
    if literal is not None:
        return literal
    sign, tokens = _tokenize(s, homophones)
    if not tokens:
        return None
    if tokens[-1] == "percent":
        return _parse_percent(sign, tokens[:-1])
    if tokens[-2:] == ["per", "cent"]:
        return _parse_percent(sign, tokens[:-2])
    if "$" in tokens or any(t in _MONEY_WORDS for t in tokens):
        return _parse_money(sign, tokens)
    if "point" in tokens:
        cut = tokens.index("point")
        int_digits = to_int_digits(tokens[:cut]) if cut else "0"
        frac = _frac_digits(tokens[cut + 1 :])
        if int_digits is None or frac is None:
            return None
        return NumValue("decimal", sign, int_digits, frac)
    if "over" in tokens or tokens[-1] in _ORDINAL_TO_DEN:
        return _parse_fraction(sign, tokens)
    int_digits = to_int_digits(tokens)
    return NumValue("integer", sign, int_digits) if int_digits is not None else None


def numeric_eq(hyp_text: str, gt_canonical: str, *, homophones: bool = False) -> bool:
    """Value-equality metric: parse the hypothesis and compare it with the canonical reference."""
    return values_equal(parse_numeric(hyp_text, homophones=homophones), parse_numeric(gt_canonical))


# ---------------------------------------------------------------------------
# hard negatives (judge audits)
# ---------------------------------------------------------------------------


def perturb_canonical(gt_canonical: str, rng: random.Random) -> str:
    """Return the canonical form with exactly one digit changed (same length and kind).

    Used as a hard negative when auditing judges for false accepts. Never returns the original and
    never introduces a leading zero in a multi-digit integer part.
    """
    value = parse_numeric(gt_canonical)
    if value is None:
        msg = f"cannot parse canonical reference: {gt_canonical!r}"
        raise ValueError(msg)
    fields = [("digits", value.digits)]
    if value.frac is not None:
        fields.append(("frac", value.frac))
    field_name, digits = fields[rng.randrange(len(fields))]
    pos = rng.randrange(len(digits))
    old = digits[pos]
    choices = [d for d in "0123456789" if d != old]
    if field_name == "digits" and pos == 0 and len(digits) > 1:
        choices = [d for d in choices if d != "0"]
    new_digits = digits[:pos] + rng.choice(choices) + digits[pos + 1 :]
    perturbed = NumValue(
        value.kind, value.sign, **{"digits": value.digits, "frac": value.frac} | {field_name: new_digits}
    )
    return perturbed.canonical()
