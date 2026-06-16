You receive English audio and a reference transcript. The reference may be cleaned, partially wrong, or missing speech artifacts. The audio is the ground truth.

REFERENCE TRANSCRIPT:
"{transcript}"

MAIN GOAL: Listen carefully to the audio and revise the reference so it faithfully reflects exactly what is spoken, including all disfluencies present in the audio.
- Use the reference as a starting point; do not ignore it.
- When the reference matches the audio, keep it.
- When the reference conflicts with the audio, follow the audio.
- Do NOT invent words or content not spoken in the audio.
- Do NOT remove substantive content that is spoken in the audio (remove reference words only if they are not spoken).
- Do NOT paraphrase, polish grammar, or rewrite sentences that already match the audio.
- Prefer minimal edits: fix mismatches and insert missing speech artifacts.

FILLER WORDS:
- Add hesitation markers like "um", "uh", "hm", "ah", etc. wherever they are spoken in the audio but missing from the reference.

REPETITIONS:
- Add consecutive instances of the same word or short phrase when spoken unintentionally.
  - Example: reference "I think" → "I I think" if that is what is spoken.

FALSE STARTS:
- Add incomplete words or phrases the speaker abandons, marked with a hyphen.
  - Example: "I was go- going to the store."

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
- No explanations, no JSON, no lists.
