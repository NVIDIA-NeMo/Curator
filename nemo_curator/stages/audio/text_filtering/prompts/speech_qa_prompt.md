You are given a transcript of a spoken audio segment in {language}. Generate at most one question-answer (Q/A) pair grounded in the transcript content.

The Q/A pair must be answerable from the transcript alone — no external knowledge, no information that is not stated or directly implied in the transcript. The question and the answer must both be in {language}, the same language as the transcript.

First decide whether the transcript is suitable for Q/A generation. The transcript is NOT suitable if:
- It is too short or carries no substantive informational content (a few words, pure filler, disfluencies only, isolated single words).
- It is a pure conversational utterance with no information content (greetings, farewells, acknowledgments, agreements, fillers like "yeah", "okay", "thanks").

For everything else (borderline cases, edge cases, content of any genre or style), use your own judgment based on whether a meaningful, well-grounded Q/A pair can plausibly be produced. When borderline, prefer SKIP over generating a weak or hallucinated pair.

Output format:

If the transcript IS suitable, output exactly two lines and nothing else:
Q: <one concrete question in {language}, not yes/no, not a trivial restatement>
A: <a concise answer in {language}, a phrase or one short sentence, directly supported by the transcript>

If the transcript is NOT suitable, output exactly the single line:
SKIP

Do not output explanations, markdown, code fences, surrounding text, blank lines before, or anything other than the two-line Q/A block or the single word SKIP.

Transcript:
{text}
