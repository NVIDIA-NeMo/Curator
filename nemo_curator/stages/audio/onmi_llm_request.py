# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains a simple stage for generating synthetic data. It takes in Empty task and a prompt and produces the output in form of a DocumentBatch.
"""

import asyncio
import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from loguru import logger

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig, LLMClient
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch


def _is_valid_url(value: str | None) -> bool:
    """Return False if value is missing, NaN, or empty string."""
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    return not (isinstance(value, str) and not value.strip())


def _is_local_path(value: str) -> bool:
    """Return True if value looks like a local file path (not http/https)."""
    if not value or not isinstance(value, str):
        return False
    s = value.strip()
    return not s.startswith(("http://", "https://"))


def _local_file_to_data_url(path: str) -> str:
    """Read a local file, encode to base64, return a data URL."""
    path_obj = Path(path).expanduser().resolve()
    if not path_obj.is_file():
        msg = "Local path is not a file: " + path
        raise FileNotFoundError(msg)
    mime_type, _ = mimetypes.guess_type(str(path_obj))
    if mime_type is None:
        mime_type = "application/octet-stream"
    raw = path_obj.read_bytes()
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def _url_or_data_url(value: str) -> str:
    """Return value as-is if remote URL; if local path, read file and return data URL."""
    if not _is_local_path(value):
        return value
    return _local_file_to_data_url(value)


def _row_to_messages(row: dict) -> list[dict]:
    """Build LLM messages from a row with optional system_text, text, image_url, audio_url."""
    messages: list[dict] = []
    if row.get("system_text"):
        messages.append({"role": "system", "content": row["system_text"]})

    content_parts: list[dict] = []
    if "image_url" in row and _is_valid_url(row.get("image_url")):
        url = _url_or_data_url(row["image_url"])
        content_parts.append({"type": "image_url", "image_url": {"url": url}})
    if "audio_url" in row and _is_valid_url(row.get("audio_url")):
        url = _url_or_data_url(row["audio_url"])
        content_parts.append({"type": "audio_url", "audio_url": {"url": url}})

    text_val = row.get("text") if "text" in row else None
    if text_val is not None and not (isinstance(text_val, float) and pd.isna(text_val)) and str(text_val).strip():
        content_parts.append({"type": "text", "text": text_val})

    if content_parts:
        messages.append({"role": "user", "content": content_parts})
    elif not messages:
        messages.append({"role": "user", "content": ""})

    return messages


@dataclass
class PrepareMessagesStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    A stage to read a jsonl file where each line is a dictionary with the following keys:
    {"system_text": "You are a helpful assistant.", "text": "Say hello to the user.", "image_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"}
    Making LLM messages with following format:
    [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"}},
        {"type": "audio_url", "audio_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"}},
        {"type": "text", "text": "What can you see and hear? Answer in one sentence."}
    ]}
    ]
    and return a DocumentBatch with the LLM messages.
    """

    name: str = "MakeMessagesStage"

    def process(self, input_batch: DocumentBatch) -> DocumentBatch:
        """Build LLM messages from each row and add a 'messages' column."""
        df = input_batch.to_pandas().copy()
        df["messages"] = df.apply(lambda row: _row_to_messages(row.to_dict()), axis=1)
        return DocumentBatch(
            data=df,
            dataset_name=input_batch.dataset_name,
            task_id=input_batch.task_id,
        )


@dataclass
class OmniLLMRequestStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    A stage for requesting LLM with messages and producing the predicted text in form of a DocumentBatch.

    Example input messages:
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"}},
        {"type": "audio_url", "audio_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"}},
        {"type": "text", "text": "What can you see and hear? Answer in one sentence."}
    ]}
    ]
    Example output predicted text:
    predicted_text = "I can see a car and hear a cough."
    It drops the 'messages' column and adds the 'predicted_text' column after processing.

    """

    client: AsyncLLMClient | LLMClient
    model_name: str
    generation_config: GenerationConfig | None = None
    name: str = "OmniLLMRequestStage"
    fields_to_drop: list[str] = field(default_factory=lambda: ["messages"])
    predicted_text_key: str = "predicted_text"

    def __post_init__(self) -> None:
        self.is_async_client = isinstance(self.client, AsyncLLMClient)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["text"]

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self.client.setup()

    def process(self, input_batch: DocumentBatch) -> DocumentBatch:
        """
        Process the input messages and produce the predicted text.
        Drop the 'messages' column and add the 'predicted_text' column after processing.
        """
        input_df = input_batch.to_pandas().copy()
        responses = self._process_async(input_df) if self.is_async_client else self._process_sync(input_df)
        input_df[self.predicted_text_key] = responses
        input_batch.data = input_df.drop(columns=self.fields_to_drop)
        return input_batch

    def _process_llm_response(self, response: list[str]) -> str:
        """Process a single response from the LLM."""
        # Extract only the generated text content (first element of the response list)
        generated_text = response[0] if response else ""

        # Some models add ** bolding for the generated text
        if "*" in generated_text:
            generated_text = generated_text.replace("*", "")

        return generated_text

    def _process_sync(self, input_df: pd.DataFrame) -> list[str]:
        """Process samples using synchronous client (sequential)."""
        batch_size = len(input_df)
        responses = []
        for i in range(batch_size):
            logger.info(f"Generating sample {i + 1}/{batch_size} (sync)...")
            messages = input_df.iloc[i]["messages"]
            response = self.client.query_model(
                model=self.model_name,
                messages=messages,
                generation_config=self.generation_config,
            )
            generated_text = self._process_llm_response(response)
            responses.append(generated_text)
        return responses

    def _process_async(self, input_df: pd.DataFrame) -> list[str]:
        """Process samples using async client (concurrent).

        This method handles both cases:
        - Normal case: No event loop exists, creates one with asyncio.run()
        - Edge case: Called from async context, runs in separate thread
        """
        try:
            # Check if we're already in an async context
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop running - this is the expected/normal case
            # Safe to use asyncio.run() which creates its own loop
            return asyncio.run(self._generate_responses_async(input_df))

        # If we get here, there's already a loop running
        # This is an edge case (e.g., Ray async actors), but we can handle it
        # by running in a new thread with its own loop
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, self._generate_responses_async(input_df))
            return future.result()

    async def _generate_responses_async(self, input_df: pd.DataFrame) -> list[str]:
        """Generate responses asynchronously using concurrent requests."""
        batch_size = len(input_df)

        async def generate_single_response(_i: int) -> str:
            messages = input_df.iloc[_i]["messages"]
            response = await self.client.query_model(
                model=self.model_name,
                messages=messages,
                generation_config=self.generation_config,
            )
            return self._process_llm_response(response)

        tasks = [generate_single_response(i) for i in range(batch_size)]
        return await asyncio.gather(*tasks)
