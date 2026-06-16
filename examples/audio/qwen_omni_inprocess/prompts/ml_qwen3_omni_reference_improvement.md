You receive {language} audio and a reference transcript. The reference may be cleaned, partially wrong, or missing speech artifacts. The audio is the ground truth.

REFERENCE TRANSCRIPT:
"{transcript}"

MAIN GOAL: Listen carefully to the audio and revise the reference so it faithfully reflects exactly what is spoken in {language}, including all disfluencies present in the audio.
- Use the reference as a starting point; do not ignore it.
- When the reference matches the audio, keep it unchanged.
- When the reference conflicts with the audio, follow the audio.
- Do NOT invent words or content not spoken in the audio.
- Do NOT remove substantive content that is spoken in the audio (remove reference words only if they are not spoken).
- Do NOT paraphrase, polish grammar, or rewrite sentences that already match the audio.
- Prefer minimal edits: fix mismatches and insert missing speech artifacts.
- Write the output in {language}, using the script and spelling natural to that language.

KEEP REFERENCE DISFLUENCIES:
- If the reference already has fillers, repetitions, false starts, colloquial or informal forms, or grammatical errors, keep them.
- Do NOT clean up or remove disfluencies that are already in the reference and are spoken in the audio.

BACKGROUND / QUIET SPEECH:
- Keep all speech in the reference that is audible in the audio, including quieter or secondary voices.
- Do NOT drop words just because they are softer or less prominent than the main speaker.

FILLER WORDS:
- Add hesitation markers and fillers natural to {language} wherever they are spoken in the audio but missing from the reference.
- Do NOT remove fillers that are already in the reference and are spoken in the audio.

REPETITIONS:
- Add consecutive instances of the same word or short phrase when spoken unintentionally.
- Do NOT remove repetitions already in the reference if they are spoken in the audio.

FALSE STARTS:
- Add incomplete words or phrases the speaker abandons, marked with a hyphen.
- Do NOT remove false starts already in the reference if they are spoken in the audio.

COLLOQUIAL / INFORMAL FORMS:
- If the reference uses a standard or formal form but the speaker used a colloquial, reduced, or informal form in {language}, use the spoken form.
- Preserve colloquial and informal forms exactly as spoken. Do NOT expand them into standard or formal written forms.

WRONG GRAMMAR:
- Keep grammatical errors as spoken. Do NOT correct grammar, agreement, tense, or other linguistic errors.

NUMERICALS:
- Keep numbers as spoken in words in {language}. Do NOT convert them to digits unless that is how they were spoken.

Output format:
- Return ONLY the revised transcription text in {language}.
- No explanations, no JSON, no lists.
