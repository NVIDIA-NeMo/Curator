You receive:
1) An audio file,
2) A Ground Truth Transcription of the audio {transcript}.

Goal: To normalize numbers from the text and add any disfluencies that are present in the audio.

ALLOWED ONLY:
1) Normalize numeric expressions into words exactly as they are SPOKEN in the audio.
- Mixed format is forbidden:
    Bad: "5 percent", "2 zeros"
    Good: "five percent", "two zeros"
- Normalize: percentages, currencies, units, ranges, decimals, dates/years — ONLY if they are spoken.
- If a unit (for example “percent”) is NOT spoken, do not add it.
2) Add any disfluencies present in the audio.
- Disfluencies as "um", "uh" that are present in the audio should be added to the text.
- If word is repeated in the audio but missing from ground truth add it to the text.

ENTITIES (names, places, brands, titles, etc.) should be the same as inGround Truth Transcription:
- Keep every named entity from the reference in its exact written form: spelling, casing, script, and punctuation. This includes names, places, brands, titles, acronyms, and other proper nouns.

OUTPUT FORMAT:
- Return only the final text.
- No explanations, no JSON, no lists.
