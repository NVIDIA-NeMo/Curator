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

from typing import Final

import pandas as pd

from nemo_curator.filters.bitext_filter import BitextFilter
from nemo_curator.filters.models.qe_models import COMETQEModel, PyMarianQEModel, QEModel
from nemo_curator.utils.decorators import batched
from nemo_curator.utils.distributed_utils import NoWorkerError, load_object_on_worker


class QualityEstimationFilter(BitextFilter):
    """(Bitext filter) Use a Quality Estimation (QE) model to score individual segments and filter based on estimated quality score.
    (reference: https://arxiv.org/pdf/2311.05350)
    """

    # a mapping from supported model names to their corresponding model class
    SUPPORTED_MODELS: Final[dict[str, type[QEModel]]] = {
        "comet-qe": COMETQEModel,
        "cometoid-wmt23": PyMarianQEModel,
        "cometoid-wmt23-mqm": PyMarianQEModel,
    }

    def __init__(self, model_name: str, cutoff: float, mode: str = "always_en_x", gpu: bool = False, **kwargs):
        """Args:
            model_name (_type_): Name of the model, as listed in the `SUPPORTED_MODELS` variable.
            cutoff (_type_): A cut-off threshold for filtering. All segments with scores lower than this threshold will be tossed away.
            mode (str, optional): See `_score_document_with_qe` for definition. Defaults to "always_en_x".
            gpu (bool, optional): Whether to use GPU. Defaults to False.

        Raises:
            NotImplementedError: If a model name outside the supported model list is passed.
        """
        super().__init__(**kwargs)

        if model_name in self.SUPPORTED_MODELS:
            self._name = model_name
        else:
            msg = f"Only the following models are currently supported: {self.SUPPORTED_MODELS.keys()!s}"
            raise NotImplementedError(msg)

        self._model_path = None
        self._mode = mode
        self._cutoff = cutoff
        self._gpu = gpu

    def _score_bitext_with_qe(  # noqa: PLR0913
        self,
        model: QEModel,
        src: pd.Series,
        tgt: pd.Series,
        src_lang: pd.Series,
        tgt_lang: pd.Series,
        mode: str = "always_en_x",
    ) -> list[float]:
        """Arrange the documents according to the inference mode, call the model to estimate translation quality.

        Args:
            model (_type_): QE model object to be called.
            src (pd.Series): data frame holding the source document.
            tgt (pd.Series): data frame holding the target document.
            src_lang (pd.Series): data frame holding the list of source languages.
            tgt_lang (pd.Series): data frame holding the list of target languages.
            mode (str, optional): Currently three inference modes are supported:

                - `simple`: Maintain the translation direction as specified in the data and
                    simply pass the corresponding fields to the quality estimation model.
                - `always_en_x`: Always pass the English side as the source and non-English side as the target.
                    This is the strategy used by the referenced paper: https://arxiv.org/pdf/2311.05350.
                - `bidi`: Estimate quality on both directions, then average the score. Potentially more accurate
                    when original translation direction is uncertain (note that "original" translation direction
                    might have been flipped while building the data), but also twice as expensive computationally.

                Defaults to "always_en_x".

        Returns:
            List[float]: A list of float scores corresponding to the individual score of each documents.
        """

        def _is_en_x(src_lang: str, tgt_lang: str) -> bool:
            return src_lang == "en" and tgt_lang != "en"

        def _has_en(src_lang: str, tgt_lang: str) -> bool:
            return src_lang == "en" and tgt_lang == "en"

        model_class = self.SUPPORTED_MODELS[self._name]

        if mode == "simple":
            model_input = [model_class.wrap_qe_input(s, t) for s, t in zip(src, tgt, strict=False)]
            return model.predict(model_input)
        elif mode == "always_en_x":
            # if English is included but it's on the target side, flip to make sure we are scoring with en-x
            # this strategy was proposed in: https://aclanthology.org/2023.wmt-1.50.pdf
            model_input = [
                model_class.wrap_qe_input(s, t, reverse=(_has_en(sl, tl) and not _is_en_x(sl, tl)))
                for s, t, sl, tl in zip(src, tgt, src_lang, tgt_lang, strict=False)
            ]
            return model.predict(model_input)
        elif mode == "bidi":
            # score twice -- once forward and once backward
            fwd_input = [model_class.wrap_qe_input(s, t) for s, t in zip(src, tgt, strict=False)]
            rev_input = [model_class.wrap_qe_input(s, t, reverse=True) for s, t in zip(src, tgt, strict=False)]
            scores = model.predict(fwd_input + rev_input)  # making one call to take advantage of batching
            # first half is forward score, second half is reverse score -- now we unpack and average
            fwd_scores = scores[: len(src)]
            rev_scores = scores[len(src) :]
            return [(fs + rs) / 2 for fs, rs in zip(fwd_scores, rev_scores, strict=False)]
        else:
            raise NotImplementedError

    @batched
    def score_bitext(self, src: pd.Series, tgt: pd.Series, src_lang: pd.Series, tgt_lang: pd.Series) -> pd.Series:
        """Wrapper function that scores documents in a data frame. Most work is done in `_score_document_with_qe`.

        Args:
            Takes two metadata fields: `src_lang` and `tgt_lang`. Refer to `_score_bitext_with_qe` function for details.

        Raises:
            RuntimeError: If input data frame arguments doesn't have the same length.

        Returns:
            pd.Series: A list of float scores corresponding to the individual score of each documents.
        """

        if not len(src) == len(tgt) == len(src_lang) == len(tgt_lang):
            msg = "Different fields of the data frame should have the same length"
            raise RuntimeError(msg)

        model_attr = f"{self._name}_{self._model_path}"
        try:
            model = load_object_on_worker(
                model_attr,
                self.SUPPORTED_MODELS[self._name].load_model,
                {"model_name": self._name, "gpu": self._gpu},
            )
        except NoWorkerError:
            return pd.Series([-1.0 for _ in range(len(src))])

        scores = self._score_bitext_with_qe(model, src, tgt, src_lang, tgt_lang, self._mode)

        return pd.Series(scores, index=src.index)

    def keep_bitext(self, score: float) -> bool:
        """Decides whether a single document should be retained according to a threshold of estimated quality score."""
        return score >= self._cutoff
