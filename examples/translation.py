from nemo_curator.synthetic.translate import TranslationDataGenerator

if __name__ == "__main__":
    # Function-based usage
    text = "Once upon a time, there were three little pig brothers..."
    generator = TranslationDataGenerator(
        base_url="http://localhost:11434/v1",
        api_key="",
        init_translate_model="gpt-oss:latest",
        reflection_model="gpt-oss:latest",
        improvement_model="gpt-oss:latest",
        hf_tokenizer="openai/gpt-oss-20b",
        hf_token="",
        temperature=1.0,
        top_p=1.0,
        max_tokens=24576,
        max_token_per_chunk=5000,
        source_lang="English",
        target_lang="Traditional Chinese",
        country="Taiwan",
    )
    translations = generator.generate(text)
    print(generator.parse_response(translations))

    # YAML-based usage (parameters only, text provided in code)
    generator_yaml = TranslationDataGenerator.from_yaml("config/translation_config.yaml")
    print(generator_yaml.generate_from_yaml("config/translation_config.yaml", text))

    # Pipeline DataFrame usage
    import pandas as pd
    df = pd.DataFrame(
        {
            "text": [
                "Once upon a time, there were three little pig brothers...",
                "The quick brown fox jumps over the lazy dog.",
            ]
        }
    )
    df_translated = generator_yaml.generate_from_dataframe(df, text_column="text", batch_size=16)
    print(df_translated.head())

    # Async test for async_generate_from_dataframe
    import asyncio
    async def test_async_generate_from_dataframe():
        df_translated = await generator_yaml.async_generate_from_dataframe(df, text_column="text", batch_size=16)
        print("[Async]\n", df_translated.head())

    asyncio.run(test_async_generate_from_dataframe())