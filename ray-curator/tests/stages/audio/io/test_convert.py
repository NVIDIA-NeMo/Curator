import pandas as pd

from ray_curator.stages.audio.io.convert import AudioToDocumentStage
from ray_curator.tasks import AudioBatch


def test_audio_to_document_stage_converts_batch() -> None:
    audio = AudioBatch(
        task_id="t1",
        dataset_name="ds",
        data=[
            {"audio_filepath": "/a.wav", "text": "hello"},
            {"audio_filepath": "/b.wav", "text": "world"},
        ],
    )

    stage = AudioToDocumentStage()
    out = stage.process(audio)

    assert isinstance(out, list) and len(out) == 1
    doc = out[0]
    assert isinstance(doc.data, pd.DataFrame)
    assert list(doc.data.columns) == ["audio_filepath", "text"]
    assert doc.task_id == audio.task_id
    assert doc.dataset_name == audio.dataset_name
