import os
import re
import unicodedata
from typing import Final

import kenlm
import sentencepiece


class SentencePiece:
    def __init__(self, model: str):
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.load(str(model))

    def tokenize(self, text: str) -> str:
        """Tokenize text using SentencePiece model"""
        return " ".join(self.sp.encode_as_pieces(text))


class KenlmModel:
    # Regex patterns and character mappings
    DIGIT_RE = re.compile(r"\d")
    UNICODE_PUNCT: Final[dict[str, str]] = {
        "，": ",",  # Chinese comma
        "。": ".",  # Chinese period
        "、": ",",  # Chinese enumeration comma
        "„": '"',  # German opening quote
        "“": '"',  # Left double quotation mark
        "”": '"',  # Right double quotation mark
        "«": '"',  # French opening quote
        "»": '"',  # French closing quote
        "１": '"',  # Fullwidth digit one
        "」": '"',  # Japanese closing quote
        "「": '"',  # Japanese opening quote
        "《": '"',  # Chinese opening quote
        "》": '"',  # Chinese closing quote
        "´": "'",  # Acute accent
        "∶": ":",  # Ratio
        "：": ":",  # Fullwidth colon
        "？": "?",  # Fullwidth question mark
        "！": "!",  # Fullwidth exclamation mark
        "（": "(",  # Fullwidth left parenthesis
        "）": ")",  # Fullwidth right parenthesis
        "；": ";",  # Fullwidth semicolon
        "–": "-",  # En dash
        "—": " - ",  # Em dash
        "．": ". ",  # Fullwidth period
        "～": "~",  # Fullwidth tilde
        "'": "'",  # Single quote
        "…": "...",  # Horizontal ellipsis
        "━": "-",  # Box drawings heavy horizontal
        "〈": "<",  # Left angle bracket
        "〉": ">",  # Right angle bracket
        "【": "[",  # Left black lenticular bracket
        "】": "]",  # Right black lenticular bracket
        "％": "%",  # Fullwidth percent sign
        "►": "-",  # Black right-pointing pointer
    }

    def __init__(  # noqa: PLR0913
        self,
        model_path: str,
        language: str,
        lower_case: bool = False,
        remove_accents: bool = False,
        normalize_numbers: bool = True,
        punctuation: int = 1,
    ):
        # Load models
        self.model = kenlm.Model(os.path.join(model_path, f"{language}.arpa.bin"))
        self.tokenizer = SentencePiece(os.path.join(model_path, f"{language}.sp.model"))

        # Store normalization settings
        self.accent = remove_accents
        self.case = lower_case
        self.numbers = normalize_numbers
        self.punct = punctuation

        # Compile regex patterns for better performance
        self.unicode_punct_re = re.compile(f"[{''.join(self.UNICODE_PUNCT.keys())}]")
        self.non_printing_chars_re = re.compile(f"[{''.join(map(chr, list(range(32)) + list(range(127, 160))))}]")

    @classmethod
    def from_pretrained(cls, model_path: str, language: str) -> "KenlmModel":
        """Factory method to create a model with default settings"""
        return cls(model_path, language)

    def get_perplexity(self, text: str, normalize: bool = True) -> float:
        """Calculate perplexity score for the given text"""
        if normalize:
            text = self.normalize(text)

        # Tokenize text
        tokenized = self.tokenizer.tokenize(text)

        # Calculate perplexity
        total_log_score, total_length = 0, 0
        for line in tokenized.split("\n"):
            log_score = self.model.score(line)
            length = len(line.split()) + 1
            total_log_score += log_score
            total_length += length

        # Convert log score to perplexity
        return round(10.0 ** (-total_log_score / total_length), 1)

    def normalize(self, text: str) -> str:
        """Apply all configured normalizations to text"""
        text = text.strip()
        if not text:
            return text

        # Apply configured normalizations
        if self.case:
            text = text.lower()
        if self.accent:
            text = self._strip_accents(text)
        if self.numbers:
            text = self.DIGIT_RE.sub("0", text)
        if self.punct == 1:
            text = self._replace_unicode_punct(text)
        elif self.punct == 2:  # noqa: PLR2004
            text = self._remove_unicode_punct(text)

        return self._remove_non_printing_chars(text)

    def _strip_accents(self, text: str) -> str:
        """Remove accents from Unicode characters"""
        normalized = unicodedata.normalize("NFD", text)
        result = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
        return result if result != text else text

    def _replace_unicode_punct(self, text: str) -> str:
        """Replace Unicode punctuation with ASCII equivalents"""
        return "".join(self.UNICODE_PUNCT.get(c, c) for c in text)

    def _remove_unicode_punct(self, text: str) -> str:
        """Remove all Unicode punctuation"""
        return self.unicode_punct_re.sub("", text)

    def _remove_non_printing_chars(self, text: str) -> str:
        """Remove non-printing characters"""
        return self.non_printing_chars_re.sub("", text)
