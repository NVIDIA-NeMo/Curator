# Language Identification

You are given a transcript of spoken audio. Your task is to identify the language(s) in which the text is written, including any code-switching.

Analyze the vocabulary, grammar, script, and word patterns. Spoken audio can be code-switched as well — a speaker may mix two or more languages in a single utterance. Detect this instead of collapsing everything to one label.

## Guidelines

- Identify the PRIMARY language: the one that carries most of the meaningful content.
- Also list EVERY language present in the transcript, including the primary one.
- Return full English names of languages (e.g. English, Spanish, French, German, Hindi, Chinese). Do not return ISO codes, locale tags, or abbreviations (e.g. do not return "en", "es", or "fr-FR").
- Proper nouns and isolated loanwords do NOT count as a separate language; only count a language if it contributes real words or phrases.
- If only one language is present, list just that one.
- If the text is too short to identify confidently, choose the most likely language(s) based on the available words and script.

## Output format

Return exactly two lines, nothing else. No explanations, quotes, or extra formatting:

```
Primary: <primary language name>
Languages: <comma-separated language names, including the primary>
```

## Examples

Transcript: "Bonjour, comment allez-vous aujourd'hui?"
Primary: French
Languages: French

Transcript: "El clima está muy agradable esta mañana."
Primary: Spanish
Languages: Spanish

Transcript: "मुझे लगता है कि हमें इस project को कल तक finish कर देना चाहिए।"
Primary: Hindi
Languages: Hindi, English

Transcript: "I think we should schedule the meeting for next Tuesday."
Primary: English
Languages: English
