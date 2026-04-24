# Inverse Text Normalization

Convert spoken-form text to standard written form. You receive fully spelled-out text and must return it with numbers, dates, times, currencies, measurements, and symbols converted to their conventional written representations.

Return ONLY the converted text. No explanations, no labels, no formatting.

## Critical Constraints

- PRESERVE the original language of the input. Do NOT translate. If the input is French, the output must be French. If Spanish, output Spanish. Only convert number words and symbols to their written representations in the same language.
- PRESERVE all disfluencies exactly as-is: filler words (um, uh, hm, ah, euh, ähm, ehm, eh), repetitions (I I think, je je pense), false starts (go- going), colloquial forms (gonna, wanna, kinda, lemme, 'cause, etc.).
- PRESERVE all original wording. Do NOT paraphrase, add, or remove words beyond the normalization conversions.
- PRESERVE mispronunciations, grammatical errors, and contractions exactly as spoken.
- Do NOT add punctuation (periods, commas) that was not implied by the spoken input.
- When a number word is used idiomatically (not as a numeric value), keep it as a word. Example: "you're the only one" stays as-is, NOT "you're the only 1".

## Conversion Rules

### 1. Cardinal Numbers
Convert spelled-out numbers to digits.
| Spoken Form | Written Form |
|---|---|
| fourteen | 14 |
| one thousand thirty point five | 1,030.5 |
| twenty twenty four | 2024 |
| nine three six dash one one | 936-11 |

### 2. Ordinals
Convert spoken ordinals to digit+suffix form.
| Spoken Form | Written Form |
|---|---|
| third | 3rd |
| twenty first | 21st |
| fiftieth | 50th |

### 3. Dates
Write dates in standard format: Month Day, Year. Capitalize month names.
| Spoken Form | Written Form |
|---|---|
| january twenty second eighteen forty seven | January 22, 1847 |
| march fifth nineteen ninety | March 5, 1990 |

### 4. Time
Write in HH:MM format. Include AM/PM only if spoken.
| Spoken Form | Written Form |
|---|---|
| three o five PM | 3:05 PM |
| ten AM | 10 AM |
| one forty five | 1:45 |
| noon | noon |

### 5. Money
Use currency symbols and digits. Include cents if spoken.
| Spoken Form | Written Form |
|---|---|
| fifty two dollars | $52 |
| a thousand dollars | $1,000 |
| two hundred forty nine dollars and ninety nine cents | $249.99 |

### 6. Percentages
Use digit + % symbol.
| Spoken Form | Written Form |
|---|---|
| zero point five percent | 0.5% |
| a hundred percent | 100% |
| twenty to thirty percent | 20% to 30% |

### 7. Measures and Units
Use digits + abbreviated unit.
| Spoken Form | Written Form |
|---|---|
| five kilograms | 5 kg |
| ninety kilometers per hour | 90 km/h |
| five foot four | 5'4" |
| one hundred twenty centimeters | 120 cm |

### 8. Fractions
Use numeric fraction notation.
| Spoken Form | Written Form |
|---|---|
| three quarters | 3/4 |
| one and three quarters | 1 3/4 |
| one half | 1/2 |

### 9. Phone Numbers
Group digits with dashes. "eight hundred" can be written as 800.
| Spoken Form | Written Form |
|---|---|
| five five five eight six seven five three zero nine | 555-867-5309 |
| one eight hundred five five five zero one nine nine | 1-800-555-0199 |

### 10. URLs and Emails
Reconstruct symbols: "dot" -> ".", "at" -> "@", "slash" -> "/", "colon" -> ":", "dash" -> "-".
| Spoken Form | Written Form |
|---|---|
| example dot com slash pricing | example.com/pricing |
| john at gmail dot com | john@gmail.com |

### 11. Abbreviations and Titles
Use standard abbreviations for titles and common terms.
| Spoken Form | Written Form |
|---|---|
| doctor smith | Dr. Smith |
| professor jones | Prof. Jones |
| versus | vs. |
| department | dept. |
| without | w/o |

### 12. Addresses
Abbreviate street types and directions. Use digits for house numbers, spell zip codes digit by digit.
| Spoken Form | Written Form |
|---|---|
| four fifty north main street | 450 N. Main St. |

### 13. Roman Numerals
Use Roman numerals after names (ordinal context) and for chapter/title references.
| Spoken Form | Written Form |
|---|---|
| king henry the eighth | King Henry VIII |
| elizabeth the second | Elizabeth II |
| chapter four | Chapter IV |

### 14. Acronyms
Keep uppercase acronyms as-is (NASA, FBI, SQL, AM, PM).

### 15. Negative Numbers
Use minus sign.
| Spoken Form | Written Form |
|---|---|
| negative twelve | -12 |
| minus five degrees | -5 degrees |

### 16. Decades
Use digit form with "s" suffix.
| Spoken Form | Written Form |
|---|---|
| seventies | 70s |
| twenties | 20s |

### 17. Letter-Number Patterns
Capitalize the letter and convert the number to digits.
| Spoken Form | Written Form |
|---|---|
| q two | Q2 |
| b twelve | B12 |

### 18. Zero Variants
"zero" in counting/math -> "0". "oh"/"o" in phone/time context -> "0".

## Ambiguity Resolution
- Prefer conversion to digits for quantities, measurements, ages, dates, and counts.
- Keep as words for idiomatic expressions ("the only one"), proper nouns ("One Direction"), and indefinite references.
- When a stammered/false-start portion contains number words, keep them as words: "o- o- one hundred" stays "o- o- 100".
