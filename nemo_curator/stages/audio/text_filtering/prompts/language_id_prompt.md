# Language Identification

You are given a transcript of spoken audio. Your task is to identify the primary language in which the text is written.

Analyze the vocabulary, grammar, script, and word patterns to determine the language. Focus on the dominant language of the transcript — the language that carries most of the meaningful content.

## Guidelines

- Return the full English name of the language (e.g. English, Spanish, French, German, Hindi, Chinese).
- Do not return ISO codes, locale tags, or abbreviations (e.g. do not return "en", "es", or "fr-FR").
- If the transcript mixes two or more languages, identify the language that accounts for the majority of the text.
- Proper nouns, loanwords, or short English insertions in an otherwise non-English transcript should not change the primary language label.
- If the text is too short to identify confidently, choose the most likely language based on the available words and script.

## Output format

Return ONLY the language name. No explanations, labels, quotes, punctuation, or extra formatting.

## Examples

Transcript: "Bonjour, comment allez-vous aujourd'hui?"
Language: French

Transcript: "El clima está muy agradable esta mañana."
Language: Spanish

Transcript: "I think we should schedule the meeting for next Tuesday."
Language: English

Transcript: "आज मौसम बहुत अच्छा है।"
Language: Hindi
