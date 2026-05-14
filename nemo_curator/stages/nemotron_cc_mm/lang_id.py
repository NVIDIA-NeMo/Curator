"""FastText ``lid.176`` English language-ID filter for ``InterleavedBatch``.

Loads the FastText language-ID model on each worker and scores the
aggregated text of every sample.  Drops samples whose top predicted
language is not English or whose confidence falls below a cutoff.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from nemo_curator.stages.nemotron_cc_mm.text_filters import (
    BaseInterleavedSampleFilterStage,
    aggregate_doc_text,
)

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

# Default location for the FastText lid.176 model on this machine.
DEFAULT_LID_PATH = Path(
    "/home/aot/codebase/nemotron_cc_mm/data/models/lid.176.bin"
)

LID_176_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
)


@dataclass
class InterleavedFastTextLangIDFilterStage(BaseInterleavedSampleFilterStage):
    """Keep only samples whose aggregated text is predicted as ``target_lang``.

    Parameters
    ----------
    model_path:
        Path to ``lid.176.bin``.  Download from
        ``https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin``.
    target_lang:
        FastText label suffix to keep (default ``"en"``).
    min_score:
        Confidence cutoff (default 0.65 — matches MINT-1T).
    """

    model_path: str = field(default_factory=lambda: str(DEFAULT_LID_PATH))
    target_lang: str = "en"
    min_score: float = 0.65
    name: str = "interleaved_fasttext_lang_id_filter"

    # Model loaded per worker.
    _model: object | None = field(default=None, init=False, repr=False)

    def setup_on_node(
        self, node_info: NodeInfo, worker_metadata: WorkerMetadata
    ) -> None:  # noqa: ARG002
        """Best-effort: warn if the model file is missing on this node."""
        if not os.path.exists(self.model_path):
            msg = (
                f"FastText lid.176 model not found at {self.model_path}. "
                f"Download with:\n"
                f"    curl -L {LID_176_URL} -o {self.model_path}\n"
                f"or call nemo_curator.stages.nemotron_cc_mm.lang_id.download_lid_176()."
            )
            raise FileNotFoundError(msg)

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        import fasttext  # local import; heavy
        self._model = fasttext.load_model(self.model_path)

    def is_sample_ok(self, sample_id: str, group: pd.DataFrame) -> bool:
        if self._model is None:
            # Defensive: setup() should have been called.
            self.setup()
        text = aggregate_doc_text(group)
        if not text:
            return False
        # FastText doesn't accept newlines in input.
        text_clean = text.replace("\n", " ").strip()
        if not text_clean:
            return False
        labels, scores = self._model.predict([text_clean], k=1)  # type: ignore[union-attr]
        # labels: [["__label__en"]], scores: [[0.97]]
        label = labels[0][0]
        score = float(scores[0][0])
        lang = label.split("__label__")[-1].lower()
        return lang == self.target_lang and score >= self.min_score


# ---------------------------------------------------------------------------
# Helper: download lid.176 to the default location.
# ---------------------------------------------------------------------------
def download_lid_176(dest: str | os.PathLike[str] | None = None) -> Path:
    """Download FastText lid.176.bin (~131 MB) if missing.  Returns dest path."""
    import urllib.request

    path = Path(dest) if dest else DEFAULT_LID_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    print(f"Downloading {LID_176_URL} -> {path}", flush=True)
    urllib.request.urlretrieve(LID_176_URL, path)
    return path
