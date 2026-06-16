You receive English audio and a reference transcript. The reference may be cleaned, partially wrong, or missing speech artifacts. The audio is the ground truth.

REFERENCE TRANSCRIPT:
"{transcript}"

MAIN GOAL: Listen carefully to the audio and revise the reference so it faithfully reflects exactly what is spoken, including all disfluencies present in the audio.
- Use the reference as a starting point; do not ignore it.
- When the reference matches the audio, keep it unchanged.
- When the reference conflicts with the audio, follow the audio.
- Do NOT invent words or content not spoken in the audio.
- Do NOT remove substantive content that is spoken in the audio (remove reference words only if they are not spoken).
- Do NOT paraphrase, polish grammar, or rewrite sentences that already match the audio.
- Prefer minimal edits: fix mismatches and insert missing speech artifacts.

KEEP REFERENCE DISFLUENCIES:
- If the reference already has fillers ("um", "uh", "hm", "ah"), repetitions, false starts, colloquial reductions, or grammatical errors, keep them.
- Do NOT clean up, normalize, or remove disfluencies that are already in the reference and are spoken in the audio.
- Only remove a disfluency from the reference if you are certain it was not spoken.

BACKGROUND / QUIET / OVERLAPPING SPEECH:
- The reference may include more than just the loudest speaker: background talk, distant speech, overlap, or a second speaker mixed in.
- Treat every part of the reference as real speech to preserve unless the audio clearly shows it was not spoken.
- Do NOT drop words or phrases because they sound like background speech, are softer, or come from a less prominent speaker.
- If you can hear that part in the audio — even quietly — keep it in the transcript.
- If background or secondary speech is audible but missing from the reference, add it.
- Do NOT shorten the transcript to only the foreground or loudest voice.

FILLER WORDS:
- Add hesitation markers like "um", "uh", "hm", "ah", etc. wherever they are spoken in the audio but missing from the reference.
- Do NOT remove fillers that are already in the reference and are spoken in the audio.

REPETITIONS:
- Add consecutive instances of the same word or short phrase when spoken unintentionally.
  - Example: reference "I think" → "I I think" if that is what is spoken.
- Do NOT remove repetitions already in the reference if they are spoken in the audio.

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
- No explanations, no JSON, no lists.
