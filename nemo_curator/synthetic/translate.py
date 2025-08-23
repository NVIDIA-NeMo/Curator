from transformers import AutoTokenizer
import re

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


    def split_long_text(self, text: str, max_tokens: int = 4096) -> list:
        sentences = self.split_sentences(text)
        chunks = []
        current_chunk = ''
        for sentence in sentences:
            token_count = self.num_tokens_in_string_hf(current_chunk + sentence)
            if token_count > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += sentence
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

if __name__ == "__main__":
    text = """Once upon a time, there were three little pig brothers. They decided to each build a house to protect themselves from the big bad wolf. The eldest pig was the laziest; he quickly built a house out of straw and lay down to rest. The second pig was a bit more diligent and built a house out of wood, which was sturdier than the straw house. The youngest pig was the smartest and most hardworking. He spent a lot of time building a solid house out of bricks. One day, the big bad wolf came! He wanted to eat the pigs. He first arrived at the eldest pig's straw house, took a deep breath—'Huff—'—and blew the house down! The eldest pig ran to the second pig's house in fright. The wolf chased after him and arrived at the wooden house. He took another deep breath—'Huff—'—and blew the wooden house down too! The two pigs quickly ran to the youngest pig's brick house. The wolf was furious and ran to the brick house, took a deep breath, and blew with all his might. But the brick house didn't move at all! The wolf blew until his face turned red, but it was no use. Finally, the wolf tried to climb down the chimney, but the youngest pig had already prepared a pot of boiling water. The wolf fell in, got scalded, and ran away, never daring to bother the pigs again. From then on, the three little pigs lived happily and safely in their sturdy brick house."""
    print("\nUsing Hugging Face tokenizer:")
    textsplitter_obj = TextSplitter(model_name="google/gemma-3-27b-it")
    chunks = textsplitter_obj.split_long_text(text, max_tokens=100)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: {chunk}")