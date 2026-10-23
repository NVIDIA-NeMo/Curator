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

"""Local test config for the audio_agent unit tests.

The repo-root ``tests/conftest.py`` declares a *session-scoped, autouse*
``shared_ray_cluster`` fixture, so every test transitively starts (and waits on)
a real Ray cluster. Most of the tests here are pure logic — redaction and token math
(``test_safety``), recipe validation (``test_validate``), resource planning
(``test_planner``), acceptance math (``test_acceptance``) — and requiring Ray for those
only makes them slow and unrunnable where Ray cannot come up. Overriding the fixture
here (nearest-conftest wins) keeps them Ray-free.

**The override's original premise no longer holds, and that cost real money.** It said
"none of them execute a pipeline"; seven modules in this directory now do
(``test_delta_execution``, ``test_smoke_bounding``, ``test_resource_hardening``,
``test_reuse``, ``test_verbs``, ``test_output_honesty``, ``test_input_identity``). With
the fixture stubbed out and no ``bootstrap_ray``, each of those runs let Ray auto-init a
cluster nobody owns: ``verbs._shutdown_owned_ray`` returns immediately because
``owned_ray_address`` is ``None``, so nothing was ever torn down. One session produced 33
cluster directories under ``/tmp/ray`` and left ~160 ``ray::StageWorker`` processes
reparented to init at ~700MB each — 75GB of a 125GB box, which then OOM'd the next run and
made every subsequent failure look like a flake.

**A teardown fixture here cannot fix it, and that was measured rather than argued.** Both
the obvious shapes were tried and both are no-ops: after ``verbs.run`` returns,
``ray.is_initialized()`` is ``False`` in the pytest process, because the Ray session belongs
to the Xenna executor, not to the driver. So ``if ray.is_initialized(): ray.shutdown()``
never fires -- session-scoped left 79 workers behind and per-test left 72, which is noise
rather than a fix. (``ray.shutdown()`` itself works fine when the driver does own the
session: 0 -> 5 -> 0 in a direct test.)

The leak is roughly one orphaned worker per pipeline run, so it belongs where the run's Ray
session is owned -- the Xenna executor teardown -- and fixing it there would fix the same
leak for a user who calls ``run`` without ``--bootstrap-ray``. Until then, clear them with
``pkill -f '^ray::'`` between sessions; a poisoned box makes unrelated tests fail as though
the code were broken.
"""

from __future__ import annotations

from collections.abc import Iterator  # noqa: TC003

import pytest


@pytest.fixture(scope="session", autouse=True)
def shared_ray_cluster() -> Iterator[str]:
    """No-op override of the repo-root Ray fixture: no cluster is started for this directory."""
    return "audio_agent-unit-tests://no-ray"
