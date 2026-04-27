# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_FASTTEXT_MODEL_URLS: dict[str, str] = {
    "lid.176.bin": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
    "lid.176.ftz": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
}
_DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/nemo_curator/fasttext")

_ISO639_3_TO_1: dict[str, str] = {
    "afr": "af", "amh": "am", "ara": "ar", "asm": "as", "aze": "az",
    "bel": "be", "ben": "bn", "bos": "bs", "bul": "bg", "cat": "ca",
    "ces": "cs", "cym": "cy", "dan": "da", "deu": "de", "ell": "el",
    "eng": "en", "est": "et", "eus": "eu", "fas": "fa", "fin": "fi",
    "fra": "fr", "gle": "ga", "glg": "gl", "guj": "gu", "hau": "ha",
    "heb": "he", "hin": "hi", "hrv": "hr", "hun": "hu", "hye": "hy",
    "ibo": "ig", "ind": "id", "isl": "is", "ita": "it", "jav": "jv",
    "jpn": "ja", "kan": "kn", "kat": "ka", "khm": "km", "kor": "ko",
    "lao": "lo", "lav": "lv", "lit": "lt", "mal": "ml", "mar": "mr",
    "mkd": "mk", "mon": "mn", "msa": "ms", "mya": "my", "nep": "ne",
    "nld": "nl", "nob": "nb", "nor": "no", "ori": "or", "pan": "pa",
    "pol": "pl", "por": "pt", "ron": "ro", "rus": "ru", "sin": "si",
    "slk": "sk", "slv": "sl", "som": "so", "spa": "es", "sqi": "sq",
    "srp": "sr", "sun": "su", "swa": "sw", "swe": "sv", "tam": "ta",
    "tel": "te", "tgl": "tl", "tha": "th", "tur": "tr", "ukr": "uk",
    "urd": "ur", "vie": "vi", "xho": "xh", "yor": "yo", "zho": "zh",
    "zul": "zu",
}


@dataclass
class FastTextLIDStage(ProcessingStage[AudioTask, AudioTask]):
    """Language identification using FastText; flags non-target-language entries.

    When an entry is flagged, ``skip_me`` is set to a descriptive string:

    - ``"Wrong language"`` — detected language differs from ``target_lang``.
    - ``"Low probability of language"`` — correct language but confidence
      below ``min_lang_prob``.
    - ``"Empty text"`` — text is blank after stripping.

    An already non-empty ``skip_me`` value is never overwritten.

    Texts with fewer than ``min_word_count`` words are passed through
    without LID filtering because FastText confidence is unreliable on
    very short inputs (especially single words).

    ``model_path`` can be:
    - A HuggingFace Hub repo ID (e.g.
      ``facebook/fasttext-language-identification``), which is downloaded
      via ``huggingface_hub``.
    - An absolute path to a local ``.bin`` or ``.ftz`` file.
    - A legacy model name (``lid.176.bin`` or ``lid.176.ftz``), which is
      downloaded to ``~/.cache/nemo_curator/fasttext/`` on first use.
    """

    model_path: str = ""
    target_lang: str = "en"
    min_lang_prob: float = 0.8
    min_word_count: int = 2
    text_key: str = "pred_text"
    skip_me_key: str = "_skip_me"
    name: str = "FastTextLID"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    _model: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.model_path:
            msg = "model_path is required for FastTextLIDStage"
            raise ValueError(msg)

    def _resolve_model_path(self) -> str:
        if os.path.isfile(self.model_path):
            return self.model_path
        if self.model_path in _FASTTEXT_MODEL_URLS:
            cache_path = os.path.join(_DEFAULT_CACHE_DIR, self.model_path)
            if os.path.isfile(cache_path):
                return cache_path
            os.makedirs(_DEFAULT_CACHE_DIR, exist_ok=True)
            url = _FASTTEXT_MODEL_URLS[self.model_path]
            logger.info(f"FastTextLIDStage: downloading {self.model_path} from {url}")
            urllib.request.urlretrieve(url, cache_path)  # noqa: S310
            return cache_path
        if "/" in self.model_path:
            try:
                from huggingface_hub import hf_hub_download

                return hf_hub_download(repo_id=self.model_path, filename="model.bin")
            except Exception as exc:
                msg = f"Failed to download '{self.model_path}' from HuggingFace Hub: {exc}"
                raise ValueError(msg) from exc
        msg = (
            f"model_path '{self.model_path}' is not a valid file path, a known model name, "
            f"or a HuggingFace repo ID.  Known names: {list(_FASTTEXT_MODEL_URLS)}"
        )
        raise ValueError(msg)

    @staticmethod
    def _parse_label(raw_label: str) -> str:
        """Extract a 2-letter ISO 639-1 language code from a fasttext label.

        Handles both the legacy format (``__label__en``) and the HuggingFace
        ``facebook/fasttext-language-identification`` format
        (``__label__eng_Latn``).
        """
        lang_part = raw_label.replace("__label__", "")
        if "_" in lang_part:
            iso3 = lang_part.split("_", 1)[0]
            return _ISO639_3_TO_1.get(iso3, iso3).lower()
        return lang_part.lower()

    def setup(self, _worker_metadata: object | None = None) -> None:
        import fasttext

        resolved = self._resolve_model_path()
        self._model = fasttext.load_model(resolved)
        logger.info(f"FastTextLIDStage: loaded model from {resolved}")

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.skip_me_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.skip_me_key]

    def _predict(self, text: str) -> tuple[str, float]:
        labels, scores = self._model.predict([text], k=1)
        return self._parse_label(labels[0][0]), scores[0][0].item()

    def _process_single(self, task: AudioTask) -> AudioTask:
        if task.data.get(self.skip_me_key, ""):
            return task
        text = task.data[self.text_key]
        if not isinstance(text, str):
            return task
        text = text.strip().replace("\n", " ")
        if not text:
            if not task.data[self.skip_me_key]:
                task.data[self.skip_me_key] = f"Empty text:{self.name}"
            return task
        if len(text.split()) < self.min_word_count:
            return task
        lang, prob = self._predict(text)
        expected = self.target_lang
        if self.source_lang_key and self.source_lang_key in task.data:
            expected = task.data[self.source_lang_key]
        if not task.data[self.skip_me_key]:
            if lang != expected.lower():
                task.data[self.skip_me_key] = f"Wrong language:{self.name}"
            elif prob < self.min_lang_prob:
                task.data[self.skip_me_key] = f"Low probability of language:{self.name}"
        return task

    def process(self, task: AudioTask) -> AudioTask:
        if self._model is None:
            logger.warning(
                f"FastTextLIDStage ({self.name}): setup() was not called before process(). "
                "Calling setup() now — check that your executor invokes setup() on each worker."
            )
            self.setup()
        return self._process_single(task)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if self._model is None:
            logger.warning(
                f"FastTextLIDStage ({self.name}): setup() was not called before process_batch(). "
                "Calling setup() now — check that your executor invokes setup() on each worker."
            )
            self.setup()
        return [self._process_single(task) for task in tasks]
