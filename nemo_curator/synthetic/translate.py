"""
The work logic and prompts here are referenced from [Github: andrewyng/translation-agent](https://github.com/andrewyng/translation-agent).
"""

import yaml
import re
from nemo_curator.synthetic.generator import SyntheticDataGenerator
from nemo_curator.synthetic.prompts import (
    INITIAL_TRANSLATION_PROMPT,
    REFLECTION_COUNTRY_TRANSLATION_PROMPT,
    REFLECTION_TRANSLATION_PROMPT,
    IMPROVE_TRANSLATION_PROMPT,
)
from nemo_curator import OpenAIClient
from openai import OpenAI
from transformers import AutoTokenizer

class TextSplitter:
    """
    Utility class for splitting text into chunks based on token count.
    """
    def __init__(self, model_name: str = "", hf_token: str = "", max_token_per_chunk: int = 4096):
        self.model_name = model_name
        self.hf_token = hf_token
        self.max_token_per_chunk = max_token_per_chunk
        # Load tokenizer for counting tokens
        if self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token, use_fast=True)
        else:
            raise ValueError("Model name is empty. Please provide a valid model name.")

    def split_sentences(self, text: str) -> list[str]:
        # Split text into sentences using punctuation
        pattern = r'([^。！？；.!?;]*[。！？；.!?;])'
        sentences = re.findall(pattern, text, flags=re.UNICODE)
        last = re.sub(pattern, '', text)
        if last:
            sentences.append(last)
        return [s.strip() for s in sentences if s.strip()]

    def num_tokens_in_string_hf(self, input_str: str) -> int:
        # Count tokens in a string using the tokenizer
        return len(self.tokenizer.encode(input_str))

    def split_long_text(self, text: str, max_tokens: int = None) -> list[str]:
        # Split long text into chunks based on max token count
        if max_tokens is None:
            max_tokens = self.max_token_per_chunk
        sentences = self.split_sentences(text)
        chunks = []
        current_chunk = ''
        for sentence in sentences:
            test_chunk = current_chunk + sentence
            token_count = self.num_tokens_in_string_hf(test_chunk)
            if token_count > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk = test_chunk
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

class TranslationDataGenerator(SyntheticDataGenerator):
    """
    Synthetic data generator for translation tasks.
    Supports both function-based and YAML-based configuration.
    """
    def __init__(
        self,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        api_key: str = "",
        init_translate_model: str = "openai/gpt-oss-20b",
        reflection_model: str = "openai/gpt-oss-20b",
        improvement_model: str = "openai/gpt-oss-20b",
        hf_tokenizer: str = "openai/gpt-oss-20b",
        hf_token: str = "",
        max_token_per_chunk: int = 5000,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 8192,
        source_lang: str = "English",
        target_lang: str = "Traditional Chinese",
        country: str = "Taiwan",
    ):
        super().__init__()
        # Initialize parameters for translation pipeline
        self.base_url = base_url
        self.api_key = api_key
        self.init_translate_model = init_translate_model
        self.reflection_model = reflection_model
        self.improvement_model = improvement_model
        self.hf_tokenizer = hf_tokenizer
        self.hf_token = hf_token
        self.max_token_per_chunk = max_token_per_chunk
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.country = country
        self.openai_client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.client = OpenAIClient(self.openai_client)
        self.text_splitter = TextSplitter(model_name=self.hf_tokenizer, hf_token=self.hf_token, max_token_per_chunk=self.max_token_per_chunk)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TranslationDataGenerator":
        """
        Create a TranslationDataGenerator instance from a YAML configuration file.
        Args:
            yaml_path: Path to the YAML configuration file.
        Returns:
            TranslationDataGenerator instance.
        """
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        # Remove 'text' key if present, only pass valid constructor args
        config = dict(config)  # Make a copy
        config.pop('text', None)
        return cls(**config)

    def generate(self, llm_prompt: str | list[str]) -> list[str]:
        """
        Main pipeline for translation generation.
        Args:
            llm_prompt: The input text to be translated (str or list of str).
        Returns:
            List of improved translations.
        """
        if isinstance(llm_prompt, str):
            chunks = self.text_splitter.split_long_text(llm_prompt, max_tokens=self.max_token_per_chunk)
        else:
            chunks = llm_prompt
        results = []
        for chunk in chunks:
            initial = self._run_init_translation(chunk)
            reflection = self._run_reflection(chunk, initial)
            improved = self._run_improve_translation(chunk, initial, reflection)
            results.append(improved)
        return results

    def generate_from_yaml(self, yaml_path: str, text: str) -> str:
        """
        Run the translation pipeline using parameters from a YAML configuration file, with text provided in code.
        Args:
            yaml_path: Path to the YAML file containing parameters.
            text: The text to translate (provided in code).
        Returns:
            Concatenated translation string.
        """
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        # Only use parameters from YAML, ignore any text key
        translations = self.generate(text)
        return self.parse_response(translations)

    def parse_response(self, llm_response: str | list[str]) -> str:
        """
        Parse the LLM response(s) into a single formatted string.
        Args:
            llm_response: List of improved translations.
        Returns:
            Concatenated translation string.
        """
        if isinstance(llm_response, list):
            return "\n".join(llm_response)
        return llm_response

    def _run_init_translation(self, text: str) -> str:
        # Run initial translation step
        prompt = INITIAL_TRANSLATION_PROMPT.format(
            source_lang=self.source_lang,
            source_text=text,
            target_lang=self.target_lang,
        )
        responses = self.client.query_model(
            model=self.init_translate_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        return responses[0]

    def _run_reflection(self, text: str, initial_translation_result: str) -> str:
        # Run reflection step to improve translation
        if self.country:
            prompt = REFLECTION_COUNTRY_TRANSLATION_PROMPT.format(
                source_lang=self.source_lang,
                source_text=text,
                target_lang=self.target_lang,
                translation_1=initial_translation_result,
                country=self.country,
            )
        else:
            prompt = REFLECTION_TRANSLATION_PROMPT.format(
                source_lang=self.source_lang,
                source_text=text,
                target_lang=self.target_lang,
                translation_1=initial_translation_result,
            )
        responses = self.client.query_model(
            model=self.reflection_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        return responses[0]

    def _run_improve_translation(self, text: str, initial_translation_result: str, reflection_result: str) -> str:
        # Run improvement step to finalize translation
        prompt = IMPROVE_TRANSLATION_PROMPT.format(
            source_lang=self.source_lang,
            source_text=text,
            target_lang=self.target_lang,
            translation_1=initial_translation_result,
            reflection=reflection_result,
        )
        responses = self.client.query_model(
            model=self.improvement_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        return responses[0]

    def generate_from_dataframe(self, df, text_column: str, batch_size: int = 32, output_column: str = "translated_text", **kwargs) -> "pd.DataFrame":
        """
        Pipeline: Translate a pandas DataFrame column in batches and return DataFrame with new column.
        Args:
            df: pandas DataFrame containing text data.
            text_column: Name of the column to translate.
            batch_size: Number of rows per batch.
            output_column: Name of the column to store translations.
            kwargs: Extra arguments for self.generate.
        Returns:
            DataFrame with new column containing translations.
        """
        import pandas as pd
        results = []
        n = len(df)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = df.iloc[start:end]
            texts = batch[text_column].tolist()
            batch_translations = self.generate(texts, **kwargs)
            results.extend(batch_translations)
        df[output_column] = results
        return df

    async def async_generate_from_dataframe(self, df, text_column: str, batch_size: int = 32, output_column: str = "translated_text", **kwargs):
        """
        Asynchronous pipeline: Translate a pandas DataFrame column in batches using async requests and return DataFrame with new column.
        Args:
            df: pandas DataFrame containing text data.
            text_column: Name of the column to translate.
            batch_size: Number of rows per batch.
            output_column: Name of the column to store translations.
            kwargs: Extra arguments for self.generate.
        Returns:
            DataFrame with new column containing translations.
        """
        import pandas as pd
        import asyncio

        async def async_generate(texts, **kwargs):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.generate, texts, **kwargs)

        n = len(df)
        tasks = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = df.iloc[start:end]
            texts = batch[text_column].tolist()
            tasks.append(async_generate(texts, **kwargs))

        results = []
        batch_translations_list = await asyncio.gather(*tasks)
        for batch_translations in batch_translations_list:
            results.extend(batch_translations)
        df[output_column] = results
        return df

# Example usage for both function and YAML config
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
        max_tokens=8192,
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