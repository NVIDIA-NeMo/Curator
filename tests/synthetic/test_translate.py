import pytest

def test_translate_function():
    from nemo_curator.synthetic.translate import TranslationDataGenerator
    from unittest.mock import patch
    text = "Once upon a time, there were three little pig brothers..."
    generator = TranslationDataGenerator(
        base_url="http://localhost:11434/v1",
        api_key="",
        init_translate_model="gpt-oss:latest",
        reflection_model="gpt-oss:latest",
        improvement_model="gpt-oss:latest",
        hf_tokenizer="openai/gpt-oss-20b",
        hf_token=None,
        temperature=1.0,
        top_p=1.0,
        max_tokens=8192,
        max_token_per_chunk=5000,
        source_lang="English",
        target_lang="Traditional Chinese",
        country="Taiwan",
    )
    with patch.object(generator, "generate", return_value=["mocked", "translated", "chunks"]) as mock_generate:
        translation_chunks = generator.generate(text)
        result = generator.parse_response(translation_chunks)
        assert isinstance(result, str)
        assert len(result) > 0
        assert result == "mocked\ntranslated\nchunks"
