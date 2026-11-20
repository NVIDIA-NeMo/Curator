# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle

from nemo_curator.utils.performance_utils import StagePerfStats


def test_measurement_identity_preserves_public_metric_contract():
    stats = StagePerfStats("stage", 1.0, 2.0, 3.0, 4, {"pages": 5.0})
    expected = {
        "stage_name": "stage",
        "process_time": 1.0,
        "actor_idle_time": 2.0,
        "input_data_size_mb": 3.0,
        "num_items_processed": 4,
        "custom_metrics": {"pages": 5.0},
    }
    assert stats.to_dict() == expected
    assert dict(stats.items()) == {
        "process_time": 1.0,
        "actor_idle_time": 2.0,
        "input_data_size_mb": 3.0,
        "num_items_processed": 4,
        "custom.pages": 5.0,
    }
    assert stats == StagePerfStats("stage", 1.0, 2.0, 3.0, 4, {"pages": 5.0})
    snapshot = pickle.loads(pickle.dumps(stats))  # noqa: S301 - trusted in-memory fixture
    stats.reset()
    assert stats._measurement_id != snapshot._measurement_id
    assert stats.process_time == 0.0
    assert snapshot.to_dict() == expected


def test_adding_stats_creates_one_new_measurement():
    first, second = StagePerfStats("stage", 1.0), StagePerfStats("stage", 2.0)
    total = first + second
    assert total.process_time == 3.0
    assert total._measurement_id not in {first._measurement_id, second._measurement_id}
