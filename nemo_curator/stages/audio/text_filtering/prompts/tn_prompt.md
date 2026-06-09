# Text Normalization ({language})

Convert the written-form transcript below in **{language}** to spoken form. **Only** apply conversions from the table (numbers, dates, times, money, units, symbols, etc.). Use **{language}** conventions; English examples show the pattern.

Return ONLY normalized text. No explanations or extra formatting.

## Constraints

- Stay in **{language}**; preserve code-switching/script mix. Do NOT translate.
- Do NOT clean up speech: keep fillers (um, uh, …), repetitions, false starts (go- going), colloquial forms (gonna, wanna), grammar errors, and all non-target wording unchanged and in order.
- Do NOT add or remove words or punctuation except as required by conversions (e.g. `@` → "at" in an email).
- Denormalize digits/symbols **only** when they match Conversion Rules. Leave idiomatic numbers, proper nouns, vague quantities, disfluencies, and text already spoken as words unchanged (e.g. keep "oh eleven" as-is; do not swap "oh" ↔ "zero" unless denormalizing a digit sequence).
- In stammers/false starts, denormalize only the final clean numeric token.

## Conversion Rules

| Category | Written | Spoken |
|---|---|---|
| Cardinal | 14 / 1,030.5 / 2024 | fourteen / one thousand thirty point five / twenty twenty four |
| Ordinal | 1st / 21st / 50th | first / twenty first / fiftieth |
| Date | January 22nd, 1847 / January 22, 1990 | january twenty second eighteen forty seven / january twenty two nineteen ninety |
| Time | 3:05 PM / 10 AM / 1:45 / noon | three o five PM / ten AM / one forty five / noon |
| Money | $52 / $249.99 | fifty two dollars / two hundred forty nine dollars and ninety nine cents |
| Percent | 0.5% / 20% to 30% | zero point five percent / twenty to thirty percent |
| Units | 5 kg / 90 km/h / 5'4" | five kilograms / ninety kilometers per hour / five foot four |
| Fractions | 1/2 / 1/3 / 2/3 / 1 3/4 | half / a third / two thirds / one and three quarters |
| Phone | 5558675309 / 18005550199 | five five five eight six seven five three zero nine / one eight hundred five five five oh one nine nine |
| URL/Email | example.com/pricing / john@gmail.com | example dot com slash pricing / john at gmail dot com |
| Titles | Dr. Smith / Prof. Jones / vs. / w/o | doctor smith / professor jones / versus / without |
| Address | 450 N. Main St. | four fifty north main street |
| Roman num. | King Henry VIII / Chapter IV | king henry the eighth / chapter four |
| Negative | -12 / -5 degrees | negative twelve / minus five degrees |
| Decades | 70s / 20s | seventies / twenties |
| Letter+num | Q2 / B12 | q two / b twelve |

Additional rules:
- Structural symbols (`.`, `@`, `/`, `:`, `-`) → spoken words in URLs, emails, phones, using **{language}** forms ("dot", "at", "slash", …).
- Acronyms (NASA, FBI, AM, PM): keep natural spoken form; do not expand unrelated abbreviations.
- Phone/time "0" → "oh" or "zero" per **{language}**; zip/house numbers → digit-by-digit when denormalizing.

## Ambiguity

- Denormalize digit/symbol form for quantities, measurements, ages, dates, and counts.
- Skip idioms, proper nouns ("One Direction"), vague quantities, temporal "quarter", and idiomatic "half".
- Convert 1/2 or 1/4 to "half" / "a quarter" only for true fractions.

Written-form transcript in {language}:
{text}
