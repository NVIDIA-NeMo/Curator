from unittest.mock import Mock, patch

from nemo_curator.backends.experimental.ray_data.adapter import RayDataStageAdapter
from nemo_curator.stages.resources import Resources


def _make_stage(ray_stage_spec=None):
    stage = Mock()
    stage.batch_size = 8
    stage.resources = Resources(cpus=2.0, gpus=1.0)
    stage.ray_stage_spec.return_value = ray_stage_spec or {}
    stage.__class__.__name__ = "MockStage"
    return stage


@patch("nemo_curator.backends.experimental.ray_data.adapter.create_actor_from_stage", return_value=Mock())
@patch("nemo_curator.backends.experimental.ray_data.adapter.calculate_concurrency_for_actors_for_stage", return_value=(1, 4))
def test_process_dataset_uses_compute_for_actor_stages(mock_concurrency, mock_create_actor):
    dataset = Mock()
    dataset.map_batches.return_value = dataset
    stage = _make_stage()

    adapter = RayDataStageAdapter(stage)
    adapter.process_dataset(dataset)

    _, kwargs = dataset.map_batches.call_args
    assert kwargs["compute"] == (1, 4)
    assert "concurrency" not in kwargs


@patch("nemo_curator.backends.experimental.ray_data.adapter.create_task_from_stage", return_value=Mock())
@patch("nemo_curator.backends.experimental.ray_data.adapter.is_actor_stage", return_value=False)
def test_process_dataset_omits_compute_for_task_stages(mock_is_actor_stage, mock_create_task):
    dataset = Mock()
    dataset.map_batches.return_value = dataset
    stage = _make_stage()

    adapter = RayDataStageAdapter(stage)
    adapter.process_dataset(dataset)

    _, kwargs = dataset.map_batches.call_args
    assert "compute" not in kwargs
    assert "concurrency" not in kwargs
