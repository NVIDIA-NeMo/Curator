from pathlib import Path

from ray_curator.tasks import AudioBatch


def test_audio_batch_accepts_dict_and_list() -> None:
    # Single dict wraps into list
    b1 = AudioBatch(data={"audio_filepath": "/x.wav"})
    assert isinstance(b1.data, list)
    assert len(b1.data) == 1

    # List passes through
    b2 = AudioBatch(data=[{"audio_filepath": "/a.wav"}, {"audio_filepath": "/b.wav"}])
    assert len(b2.data) == 2


def test_audio_batch_validation_uses_filepath_key(tmp_path: Path) -> None:
    existing = tmp_path / "ok.wav"
    existing.write_bytes(b"fake")

    missing = tmp_path / "missing.wav"

    batch = AudioBatch(
        data=[{"audio_filepath": existing.as_posix()}, {"audio_filepath": missing.as_posix()}],
        filepath_key="audio_filepath",
    )

    # validate_item should be True for existing, False for missing
    assert batch.validate_item(batch.data[0]) is True
    assert batch.validate_item(batch.data[1]) is False

    # overall validate should be False (since one item is missing)
    assert batch.validate() is False
