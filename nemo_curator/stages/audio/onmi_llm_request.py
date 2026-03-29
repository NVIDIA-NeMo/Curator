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


def _is_valid_value(value: str | None) -> bool:
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


def _parse_dali_webdataset_index(index_path: Path) -> dict[str, tuple[int, int]]:
    """Parse a DALI ``wds2idx.py`` v1.2 index into ``member_name -> (data_offset, size)``.

    Each non-header line lists components as
    ``ext offset size name`` (repeated per component in the sample), matching
    `NVIDIA DALI tools/wds2idx.py <https://github.com/NVIDIA/DALI/blob/main/tools/wds2idx.py>`_.
    """
    raw = index_path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        msg = "DALI index file is empty: " + str(index_path)
        raise ValueError(msg)
    header_parts = lines[0].split()
    min_header_parts = 2
    if len(header_parts) < min_header_parts or not header_parts[0].startswith("v"):
        msg = "Invalid DALI webdataset index header: " + lines[0]
        raise ValueError(msg)
    out: dict[str, tuple[int, int]] = {}
    for line in lines[1:]:
        parts = line.split()
        n = len(parts)
        if n % 4 != 0:
            msg = (
                "Unparseable DALI index line (expected a multiple of 4 tokens "
                "ext/offset/size/name per component): " + line[:200]
            )
            raise ValueError(msg)
        for i in range(0, n, 4):
            _ext, off_s, sz_s, name = parts[i], parts[i + 1], parts[i + 2], parts[i + 3]
            out[name] = (int(off_s), int(sz_s))
    if not out:
        msg = "DALI index contains no member entries: " + str(index_path)
        raise ValueError(msg)
    return out


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

    Args:
        format: How to encode audio content parts.
            ``"audio_url"`` (default) — vLLM / Qwen3-Omni format.
            ``"input_audio"`` — OpenAI-compatible hosted API format
            (e.g. NVIDIA inference API with Gemini).
        input_tar: Path to a WebDataset tar shard. When set together with
            ``input_index``, ``audio_filepath_key`` / ``image_filepath_key`` values
            are interpreted as **member names** inside the tar (as listed by
            ``tar tf``), matching names stored in the DALI index from ``wds2idx.py``.
        input_index: Path to the DALI v1.2 index file for ``input_tar``
            (see ``tools/wds2idx.py`` in the DALI repo).
    """

    name: str = "MakeMessagesStage"
    format: str = "data_url"
    audio_filepath_key: str = "audio_filepath"
    image_filepath_key: str = "image_url"
    input_tar: str = ""
    input_index: str = ""
    user_prompt: str = "text"
    system_prompt: str = "system_text"

    _tar_member_index: dict[str, tuple[int, int]] | None = field(default=None, init=False, repr=False)

    def _tar_index_configured(self) -> bool:
        tar_set = bool(self.input_tar.strip())
        index_set = bool(self.input_index.strip())
        if tar_set ^ index_set:
            msg = "input_tar and input_index must both be set, or both empty."
            raise ValueError(msg)
        return tar_set and index_set

    def _tar_member_index_map(self) -> dict[str, tuple[int, int]]:
        if self._tar_member_index is None:
            idx_path = Path(self.input_index.strip()).expanduser().resolve()
            self._tar_member_index = _parse_dali_webdataset_index(idx_path)
        return self._tar_member_index

    def _resolve_tar_member_name(self, key: str) -> str:
        """Map a row value to the tar member name recorded in the DALI index."""
        key = key.strip()
        m = self._tar_member_index_map()
        if key in m:
            return key
        basename = Path(key).name
        matches = [n for n in m if n == key or n.endswith("/" + key) or Path(n).name == basename]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            msg = f"Ambiguous tar member key {key!r}; matches include {matches[:8]!r}"
            raise ValueError(msg)
        msg = f"Member not found in DALI index for key {key!r}"
        raise FileNotFoundError(msg)

    def _read_bytes_from_tar(self, member_key: str) -> tuple[bytes, str]:
        """Read raw file bytes from ``input_tar`` using the DALI index; return (data, path_for_mime)."""
        tar_path = Path(self.input_tar.strip()).expanduser().resolve()
        if not tar_path.is_file():
            msg = "input_tar is not a file: " + str(tar_path)
            raise FileNotFoundError(msg)
        name = self._resolve_tar_member_name(member_key)
        offset, size = self._tar_member_index_map()[name]
        with tar_path.open("rb") as f:
            f.seek(offset)
            raw = f.read(size)
        if len(raw) != size:
            msg = f"Short read from tar for {name!r}: got {len(raw)} bytes, expected {size}"
            raise OSError(msg)
        return raw, name

    def _local_file_to_data_url(self, path: str) -> tuple[str, str]:
        """Read a local file, encode to base64, return a data URL."""
        if self._tar_index_configured():
            raw, resolved_name = self._read_bytes_from_tar(path)
            mime_type, _ = mimetypes.guess_type(resolved_name)
            if mime_type is None:
                mime_type = "application/octet-stream"
            b64 = base64.standard_b64encode(raw).decode("ascii")
            return b64, mime_type

        path_obj = Path(path).expanduser().resolve()
        if not path_obj.is_file():
            msg = "Local path is not a file: " + path
            raise FileNotFoundError(msg)
        mime_type, _ = mimetypes.guess_type(str(path_obj))
        if mime_type is None:
            mime_type = "application/octet-stream"
        raw = path_obj.read_bytes()
        b64 = base64.standard_b64encode(raw).decode("ascii")
        return b64, mime_type

    def _local_file_to_input_audio(self, path: str) -> tuple[str, str]:
        """Read a local audio file, return an ``input_audio`` content part.

        The ``input_audio`` format is used by OpenAI-compatible hosted APIs
        (e.g. NVIDIA inference API with Gemini) as opposed to ``audio_url``
        which is used by vLLM.
        """
        if self._tar_index_configured():
            raw, resolved_name = self._read_bytes_from_tar(path)
            ext = Path(resolved_name).suffix.lstrip(".").lower()
            if not ext:
                ext = "wav"
            b64 = base64.standard_b64encode(raw).decode("ascii")
            return b64, ext

        path_obj = Path(path).expanduser().resolve()
        if not path_obj.is_file():
            msg = "Local path is not a file: " + path
            raise FileNotFoundError(msg)
        raw = path_obj.read_bytes()
        b64 = base64.standard_b64encode(raw).decode("ascii")
        # Derive format from extension (wav, mp3, flac, etc.)
        ext = path_obj.suffix.lstrip(".").lower()
        if not ext:
            ext = "wav"
        return b64, ext

    def _url_or_data_url(self, value: str, content_format: str) -> dict:
        """Return value as-is if remote URL; if local path, read file and return data URL.
        Args:
            value: The value to encode.
            content_format: How to encode audio content parts.
                    ``"audio_url"`` — vLLM / Qwen3-Omni format (default).
                    ``"input_audio"`` — OpenAI-compatible hosted API format
                    (e.g. NVIDIA inference API with Gemini).

        """
        if not _is_local_path(value):
            return {"type": content_format, content_format: {"url": value}}

        if content_format == "input_audio":
            b64, ext = self._local_file_to_input_audio(value)
            return {"type": content_format, content_format: {"data": b64, "format": ext}}
        elif content_format == "audio_url":
            b64, mime_type = self._local_file_to_data_url(value)
            return {"type": content_format, content_format: {"url": f"data:{mime_type};base64,{b64}"}}
        elif content_format == "input_image":
            b64, ext = self._local_file_to_input_audio(value)  # TODO: check this
            return {"type": content_format, content_format: {"data": b64, "format": ext}}
        elif content_format == "image_url":
            b64, mime_type = self._local_file_to_data_url(value)
            return {"type": content_format, content_format: {"url": f"data:{mime_type};base64,{b64}"}}
        else:
            msg = f"Invalid format: {content_format}. Supported formats: input_audio, audio_url, image_url."
            raise ValueError(msg)

    def _user_content_format_for_media(self, *, image: bool) -> str:
        """Map stage ``format`` to API content part type for image or audio."""
        if self.format == "data_url":
            return "image_url" if image else "audio_url"
        if self.format == "input_data":
            return "input_image" if image else "input_audio"
        msg = f"Invalid format: {self.format}. Supported formats: input_data, data_url."
        raise ValueError(msg)

    def _row_to_messages(self, row: dict) -> list[dict]:
        """Build LLM messages from a row with optional system_text, text, image_url, audio_url.

        Args:
            row: A dictionary with optional keys: system_text, text, image_url, audio_url.
        """
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        content_parts: list[dict] = []
        if self.image_filepath_key in row and _is_valid_value(row.get(self.image_filepath_key)):
            content_fmt = self._user_content_format_for_media(image=True)
            msg = self._url_or_data_url(row[self.image_filepath_key], content_format=content_fmt)
            content_parts.append(msg)

        if self.audio_filepath_key in row and _is_valid_value(row.get(self.audio_filepath_key)):
            content_fmt = self._user_content_format_for_media(image=False)
            msg = self._url_or_data_url(row[self.audio_filepath_key], content_format=content_fmt)
            content_parts.append(msg)

        text_val = self.user_prompt
        if text_val is not None and not (isinstance(text_val, float) and pd.isna(text_val)) and str(text_val).strip():
            content_parts.append({"type": "text", "text": text_val})

        if content_parts:
            messages.append({"role": "user", "content": content_parts})
        elif not messages:
            messages.append({"role": "user", "content": ""})

        return messages

    def process(self, input_batch: DocumentBatch) -> DocumentBatch:
        """Build LLM messages from each row and add a 'messages' column."""
        df = input_batch.to_pandas().copy()
        df["messages"] = df.apply(lambda row: self._row_to_messages(row.to_dict()), axis=1)
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
