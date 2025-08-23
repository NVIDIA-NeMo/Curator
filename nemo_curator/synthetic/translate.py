from transformers import AutoTokenizer
import re
from openai import OpenAI
from nemo_curator import OpenAIClient
from nemo_curator.synthetic.prompts import (
    INITIAL_TRANSLATION_PROMPT,
    REFLECTION_COUNTRY_TRANSLATION_PROMPT,
    REFLECTION_TRANSLATION_PROMPT,
    IMPROVE_TRANSLATION_PROMPT
)

class TextSplitter:
    def __init__(self, model_name:str="", hf_token:str="", max_token_per_chunk:int=4096):
        self.model_name = model_name
        self.hf_token = hf_token
        self.max_token_per_chunk = max_token_per_chunk
        try:
            if self.model_name != "":
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token, use_fast=True)
            else:
                raise ValueError("Model name is empty. Please provide a valid model name.")
        except Exception as e:
            raise ValueError(f"Error loading tokenizer\n{e}")

    def split_sentences(self, text: str) -> list:
        pattern = r'([^。！？；.!?;]*[。！？；.!?;])'
        sentences = re.findall(pattern, text, flags=re.UNICODE)
        last = re.sub(pattern, '', text)
        if last:
            sentences.append(last)
        return [s.strip() for s in sentences if s.strip()]

    def num_tokens_in_string_hf(self, input_str: str) -> int:
        num_tokens = len(self.tokenizer.encode(input_str))
        return num_tokens


    def split_long_text(self, text: str, max_tokens: int=None) -> list:
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

class FormatAgent():
    def __init__(self,
                 base_url:str="https://integrate.api.nvidia.com/v1",
                 api_key:str="",
                 model:str="",
                 temperature:float=1.0,
                 top_p:float=1.0,
                 max_tokens:int=8192,
                 prompt_tmp:str="",
                 ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.openai_client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        self.client = OpenAIClient(self.openai_client)
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.prompt_tmp = prompt_tmp

    def run(self, input:dict):
        prompt = self.prompt_tmp.format(**input)
        if len(prompt) == 0:
            raise ValueError("Prompt template is empty after formatting. Please provide a valid prompt template and input data.")
        responses = self.client.query_model(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens, 
        )
        return responses[0]

class TranslationWorkflow():
    def __init__(self,
                 base_url:str="https://integrate.api.nvidia.com/v1",
                 api_key:str="",
                 init_translate_model:str="openai/gpt-oss-20b",
                 reflection_model:str="openai/gpt-oss-20b",
                 improvement_model:str="openai/gpt-oss-20b",
                 hf_tokenizer:str="openai/gpt-oss-20b",
                 hf_token:str="",
                 max_token_per_chunk:int=5000,
                 temperature:float=1.0,
                 top_p:float=1.0,
                 max_tokens:int=8192,
                 source_lang:str="English",
                 target_lang:str="Traditional Chinese",
                 country:str="Taiwan",
                 ):
        self.initial_translation_agent = FormatAgent(base_url=base_url, api_key=api_key, model=init_translate_model, prompt_tmp=INITIAL_TRANSLATION_PROMPT)
        if country:
            self.reflection_agent = FormatAgent(base_url=base_url, api_key=api_key, model=reflection_model, prompt_tmp=REFLECTION_COUNTRY_TRANSLATION_PROMPT)
        else:
            self.reflection_agent = FormatAgent(base_url=base_url, api_key=api_key, model=reflection_model, prompt_tmp=REFLECTION_TRANSLATION_PROMPT)
        self.improve_translation_agent = FormatAgent(base_url=base_url, api_key=api_key, model=improvement_model, prompt_tmp=IMPROVE_TRANSLATION_PROMPT)
        self.hf_tokenizer = hf_tokenizer
        self.max_token_per_chunk = max_token_per_chunk
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.country = country
        self.hf_token = hf_token

    def demo_run(self, text:str):
        chunks = TextSplitter(model_name=self.hf_tokenizer, hf_token=self.hf_token, max_token_per_chunk=self.max_token_per_chunk).split_long_text(text, max_tokens=self.max_token_per_chunk)
        translated_chunks = []
        for chunk in chunks:
            initial_translation_result = self.initial_translation_agent.run(
                input = {
                    "source_lang": self.source_lang,
                    "source_text": chunk,
                    "target_lang": self.target_lang,
                }
            )
            print(f"initial_translation_result:\n{initial_translation_result}")
            reflection_result = self.reflection_agent.run(
                input = {
                    "source_lang": self.source_lang,
                    "source_text": chunk,
                    "target_lang": self.target_lang,
                    "translation_1": initial_translation_result,
                    "country": self.country if self.country else "N/A",
                }
            )
            print(f"reflection_result:\n{reflection_result}")
            improve_translation_result = self.improve_translation_agent.run(
                input = {
                    "source_lang": self.source_lang,
                    "source_text": chunk,
                    "target_lang": self.target_lang,
                    "translation_1": initial_translation_result,
                    "reflection": reflection_result,
                }
            )
            print(f"improve_translation_result:\n{improve_translation_result}")
            translated_chunks.append(improve_translation_result)
        return format("\n".join(translated_chunks))
    
    def run(self, text:str):
        chunks = TextSplitter(model_name=self.hf_tokenizer, hf_token=self.hf_token, max_token_per_chunk=self.max_token_per_chunk).split_long_text(text, max_tokens=self.max_token_per_chunk)
        translated_chunks = []
        for chunk in chunks:
            initial_translation_result = self.initial_translation_agent.run(
                input = {
                    "source_lang": self.source_lang,
                    "source_text": chunk,
                    "target_lang": self.target_lang,
                }
            )
            reflection_result = self.reflection_agent.run(
                input = {
                    "source_lang": self.source_lang,
                    "source_text": chunk,
                    "target_lang": self.target_lang,
                    "translation_1": initial_translation_result,
                    "country": self.country if self.country else "N/A",
                }
            )
            improve_translation_result = self.improve_translation_agent.run(
                input = {
                    "source_lang": self.source_lang,
                    "source_text": chunk,
                    "target_lang": self.target_lang,
                    "translation_1": initial_translation_result,
                    "reflection": reflection_result,
                }
            )
            translated_chunks.append(improve_translation_result)
        return format("\n".join(translated_chunks))

if __name__ == "__main__":
    text = """Once upon a time, there were three little pig brothers. They decided to each build a house to protect themselves from the big bad wolf. The eldest pig was the laziest; he quickly built a house out of straw and lay down to rest. The second pig was a bit more diligent and built a house out of wood, which was sturdier than the straw house. The youngest pig was the smartest and most hardworking. He spent a lot of time building a solid house out of bricks. One day, the big bad wolf came! He wanted to eat the pigs. He first arrived at the eldest pig's straw house, took a deep breath—'Huff—'—and blew the house down! The eldest pig ran to the second pig's house in fright. The wolf chased after him and arrived at the wooden house. He took another deep breath—'Huff—'—and blew the wooden house down too! The two pigs quickly ran to the youngest pig's brick house. The wolf was furious and ran to the brick house, took a deep breath, and blew with all his might. But the brick house didn't move at all! The wolf blew until his face turned red, but it was no use. Finally, the wolf tried to climb down the chimney, but the youngest pig had already prepared a pot of boiling water. The wolf fell in, got scalded, and ran away, never daring to bother the pigs again. From then on, the three little pigs lived happily and safely in their sturdy brick house."""
    print("\nUsing Hugging Face tokenizer:")
    textsplitter_obj = TextSplitter(model_name="openai/gpt-oss-20b")
    chunks = textsplitter_obj.split_long_text(text, max_tokens=100)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: {chunk}")

    translation_workflow = TranslationWorkflow(
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
    translation_workflow.demo_run(text)
    