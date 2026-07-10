You receive English audio and a reference transcript. The reference may be cleaned, partially wrong, or missing speech artifacts. The audio is the ground truth.

REFERENCE TRANSCRIPT:
"{transcript}"

MAIN GOAL: Listen carefully to the audio and revise the reference so it faithfully reflects exactly what is spoken, including all disfluencies present in the audio.
- Use the reference as a starting point; do not ignore it.
- When the reference matches the audio, keep it unchanged.
- When the reference conflicts with the audio, follow the audio.
- Do NOT invent words or content not spoken in the audio.
- Do NOT remove substantive content that is spoken in the audio (remove reference words only if they are not spoken).
- Do NOT paraphrase, polish grammar or rewrite sentences that already match the audio.
- Prefer minimal edits: fix mismatches and insert missing speech artifacts.
- Preserve named entities from the reference in their exact written form.
- Normalize numbers to their written form.

ENTITIES (names, places, brands, titles, etc.):
- Keep every named entity from the reference in its exact written form: spelling, casing, script, and punctuation. This includes names, places, brands, titles, acronyms, and other proper nouns.
- Do not ever transliterate, translate, re-spell, normalize, or "correct" an entity into another script.
- If enetities are part code switched data it should stay the same.

KEEP REFERENCE DISFLUENCIES:
- If the reference already has fillers, repetitions, false starts, colloquial reductions, or grammatical errors, keep them.
- Add hesitation markers and fillers natural to English wherever they are spoken in the audio but missing from the reference.
- Do NOT clean up, normalize, or remove disfluencies that are already in the reference and are spoken in the audio.
- Add consecutive instances of the same word or short phrase when spoken unintentionally.
  - Example: reference "I think" → "I I think" if that is what is spoken.


BACKGROUND / QUIET / OVERLAPPING SPEECH:
- Keep all audible speech in the reference, including quieter, distant, or overlapping voices — not just the loudest speaker.
- Add background or secondary speech that is audible but missing; do not drop words because they sound like background.

FALSE STARTS:
- Add incomplete words or phrases the speaker abandons, marked with a hyphen.
  - Example: "I was go- going to the store."
- Do NOT remove false starts already in the reference if they are spoken in the audio.

COLLOQUIAL REDUCTIONS:
- If the reference uses standard forms but the speaker used reductions, use the spoken form: "want to" → "wanna", "going to" → "gonna", etc.
- Preserve forms such as "wanna", "gonna", "kinda", "lemme", "lotta", "outta", "Imma", "sorta", "ya", "m'kay", "finna", "tryna", etc. Do NOT expand them.

WRONG GRAMMAR:
- Keep grammatical errors as spoken. Do NOT correct subject-verb agreement, tense errors, or other grammar issues.

NUMERICALS:
- Keep numbers as spoken in words. Do NOT convert them to digits.
  - Example: keep "oh eleven" or "zero eleven".

Output format:
- Return ONLY the revised transcription text.
