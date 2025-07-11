from abc import ABC, abstractmethod


class HTMLExtractorAlgorithm(ABC):
    NON_SPACED_LANGUAGES = frozenset(["THAI", "CHINESE", "JAPANESE", "KOREAN"])

    @abstractmethod
    def extract_text(self, html: str, stop_words: frozenset[str], language: str) -> list[str] | None:
        pass
