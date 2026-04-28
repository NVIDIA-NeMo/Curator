# Inverse Text Normalization

Convert spoken-form text to standard written form: numbers, dates, times, currencies, measurements, and symbols become their conventional written representations.

Return ONLY the converted text. No explanations, labels, or extra formatting.

## Constraints

- PRESERVE the input language. Do NOT translate.
- PRESERVE all original wording, disfluencies (um, uh, euh, ähm), repetitions, false starts (go- going), colloquial forms (gonna, 'cause), mispronunciations, and grammatical errors.
- Do NOT add punctuation that wasn't implied by the input.
- Do NOT paraphrase, add, or remove words beyond the conversions below.
- When a number word is idiomatic (not numeric), keep it as a word (see Ambiguity Resolution).

## Conversion Rules

| Category | Spoken | Written |
|---|---|---|
| Cardinal | fourteen / one thousand thirty point five / twenty twenty four | 14 / 1,030.5 / 2024 |
| Ordinal | first / twenty first / fiftieth | 1st / 21st / 50th |
| Date | january twenty second eighteen forty seven / january twenty two nineteen ninety | January 22nd, 1847 / January 22, 1990 |
| Time | three o five PM / ten AM / one forty five / noon | 3:05 PM / 10 AM / 1:45 / noon |
| Money | fifty two dollars / two hundred forty nine dollars and ninety nine cents | $52 / $249.99 |
| Percent | zero point five percent / twenty to thirty percent | 0.5% / 20% to 30% |
| Units | five kilograms / ninety kilometers per hour / five foot four | 5 kg / 90 km/h / 5'4" |
| Fractions | half / a third / two thirds / one and three quarters | 1/2 / 1/3 / 2/3 / 1 3/4 |
| Phone | five five five eight six seven five three zero nine / one eight hundred five five five oh one nine nine | 5558675309 / 18005550199 |
| URL/Email | example dot com slash pricing / john at gmail dot com | example.com/pricing / john@gmail.com |
| Titles | doctor smith / professor jones / versus / without | Dr. Smith / Prof. Jones / vs. / w/o |
| Address | four fifty north main street | 450 N. Main St. |
| Roman num. | king henry the eighth / chapter four | King Henry VIII / Chapter IV |
| Negative | negative twelve / minus five degrees | -12 / -5 degrees |
| Decades | seventies / twenties | 70s / 20s |
| Letter+num | q two / b twelve | Q2 / B12 |

Additional rules:
- Symbols: "dot"→".", "at"→"@", "slash"→"/", "colon"→":", "dash"→"-".
- Acronyms (NASA, FBI, SQL, AM, PM): keep uppercase as-is.
- "zero" → "0"; "oh"/"o" in phone/time contexts → "0".
- Zip codes: spell digit by digit. House numbers: digits.

## Ambiguity Resolution

- Prefer digits for quantities, measurements, ages, dates, counts.
- Keep as WORDS for idioms, proper nouns, and pronominal/indefinite uses:
  - "one of the best", "the only one", "one another", "one by one"
  - "One Direction" (proper noun), "a couple of things" (vague quantity)
- "quarter" in temporal/financial contexts stays as "quarter" ("fourth quarter", "quarter over quarter"). Only convert to 1/4 when it's a true fraction ("a quarter of a cup" → "1/4 of a cup").
- "half" → 1/2 only as a fraction; idiomatic use stays a word ("half the time", "half asleep").
- In stammers/false starts, keep number words as words: "o- o- one hundred" → "o- o- 100" (digits only on the final clean token).
