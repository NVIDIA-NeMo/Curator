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
"""Unit tests for the worker-side resumability client helpers (actor lookup,
no-op when inactive, delta fire, completed-source lookup). ``ray`` is mocked,
so no live cluster is needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from nemo_curator.utils import resumability_client as rc


class TestActorLookup:
    def test_none_when_ray_not_initialized(self) -> None:
        with patch.object(rc, "ray") as ray:
            ray.is_initialized.return_value = False
            assert rc._resumability_actor() is None
            assert rc.is_resumability_actor_active() is False
            ray.get_actor.assert_not_called()

    def test_none_when_no_actor_registered(self) -> None:
        with patch.object(rc, "ray") as ray:
            ray.is_initialized.return_value = True
            ray.get_actor.side_effect = ValueError("no such actor")
            assert rc._resumability_actor() is None
            assert rc.is_resumability_actor_active() is False

    def test_returns_handle_when_registered(self) -> None:
        with patch.object(rc, "ray") as ray:
            ray.is_initialized.return_value = True
            handle = MagicMock()
            ray.get_actor.return_value = handle
            assert rc._resumability_actor() is handle
            assert rc.is_resumability_actor_active() is True
            ray.get_actor.assert_called_with(name=rc.ACTOR_NAME, namespace=rc.ACTOR_NAME)


class TestFlushDeltas:
    def test_fires_when_active_and_nonempty(self) -> None:
        with patch.object(rc, "ray") as ray:
            ray.is_initialized.return_value = True
            handle = MagicMock()
            ray.get_actor.return_value = handle
            deltas = [("t0", "s0", 1), ("t1", "s0", -1)]
            rc.flush_resumability_deltas(deltas)
            handle.apply_deltas.remote.assert_called_once_with(deltas)

    def test_noop_when_no_deltas(self) -> None:
        with patch.object(rc, "ray") as ray:
            ray.is_initialized.return_value = True
            handle = MagicMock()
            ray.get_actor.return_value = handle
            rc.flush_resumability_deltas([])
            handle.apply_deltas.remote.assert_not_called()

    def test_noop_when_inactive(self) -> None:
        with patch.object(rc, "ray") as ray:
            ray.is_initialized.return_value = False
            # Must not raise even though there are deltas to send.
            rc.flush_resumability_deltas([("t0", "s0", 1)])


class TestSkipCompletedSources:
    def test_returns_completed_subset(self) -> None:
        with patch.object(rc, "ray") as ray:
            ray.is_initialized.return_value = True
            handle = MagicMock()
            ray.get_actor.return_value = handle
            ray.get.return_value = [True, False, True]
            assert rc.completed_resumability_sources(["a", "b", "c"]) == {"a", "c"}
            handle.are_completed.remote.assert_called_once_with(["a", "b", "c"])

    def test_empty_when_inactive(self) -> None:
        with patch.object(rc, "ray") as ray:
            ray.is_initialized.return_value = False
            assert rc.completed_resumability_sources(["a"]) == set()

    def test_empty_when_no_sources(self) -> None:
        with patch.object(rc, "ray") as ray:
            ray.is_initialized.return_value = True
            ray.get_actor.return_value = MagicMock()
            assert rc.completed_resumability_sources([]) == set()
            ray.get.assert_not_called()
