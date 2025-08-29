from ray_curator.stages.audio.metrics.get_wer import (
    GetPairwiseWerStage,
    get_cer,
    get_charrate,
    get_wer,
    get_wordrate,
)
from ray_curator.tasks import AudioBatch


def test_get_wer_basic() -> None:
    assert get_wer("a b c", "a x c") == 33.33


def test_get_cer_basic() -> None:
    assert get_cer("abc", "axc") == 33.33


def test_rates() -> None:
    assert get_charrate("abcd", 2.0) == 2.0
    assert get_wordrate("a b c d", 2.0) == 2.0


def test_pairwise_wer_stage() -> None:
    stage = GetPairwiseWerStage()
    entry = {"text": "a b c", "pred_text": "a x c"}
    out = stage.process(AudioBatch(data=[entry]))
    assert len(out) == 1
    assert out[0].data[0]["wer"] == 33.33


