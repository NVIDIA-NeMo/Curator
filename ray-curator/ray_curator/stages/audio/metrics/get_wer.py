from dataclasses import dataclass

import editdistance

from ray_curator.stages.audio.common import LegacySpeechStage
from ray_curator.tasks import AudioBatch


def get_wer(text: str, pred_text: str) -> float:
    text_words = text.split()
    pred_text_words = pred_text.split()
    word_dist = editdistance.eval(text_words, pred_text_words)

    num_words = len(text_words)
    return round(word_dist / num_words * 100.0, 2)


def get_cer(text: str, pred_text: str) -> float:
    char_dist = editdistance.eval(text, pred_text)
    num_chars = len(text)
    return round(char_dist / num_chars * 100.0, 2)


def get_charrate(text: str, duration: float) -> float:
    num_chars = len(text)
    return round(num_chars / duration, 2)


def get_wordrate(text: str, duration: float) -> float:
    num_words = len(text.split())
    return round(num_words / duration, 2)


@dataclass
class GetPairwiseWerStage(LegacySpeechStage):
    """Count pairwise word-error-rate (WER) * 100% for each pair of text and pred_text.

    WER is measured between ``data[self.text_key]`` and ``data[self.pred_text_key]``.


    Args:
        text_key (str): a string indicating which key of the data entries
            should be used to find the utterance transcript. Defaults to "text".
        pred_text_key (str): a string indicating which key of the data entries
            should be used to access the ASR predictions. Defaults to "pred_text".

    Returns:
         The same data as in the input manifest with wer_key and corresponding values.
    """

    text_key: str = "text"
    pred_text_key: str = "pred_text"
    wer_key: str = "wer"

    def process_dataset_entry(self, data_entry: dict) -> list[AudioBatch]:
        wer = get_wer(data_entry[self.text_key], data_entry[self.pred_text_key])
        data_entry[self.wer_key] = wer
        return [AudioBatch(data=data_entry)]
