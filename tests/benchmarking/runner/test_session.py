from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking"))

from runner.session import Session

_RESULTS_PATH = str(Path(__file__).resolve().parent)


def _config(entries: list[dict], **overrides: object) -> dict:
    return {
        "paths": [{"name": "results_path", "host_path": _RESULTS_PATH}],
        "entries": entries,
        **overrides,
    }


def test_session_defaults_max_timeout_s() -> None:
    session = Session.from_dict(_config([{"name": "entry_a", "script": "benchmark.py"}]))

    assert session.max_timeout_s == 14340
    assert session.entries[0].timeout_s == 7200


def test_session_rejects_timeout_above_max_timeout_s() -> None:
    with pytest.raises(ValueError, match=r"entry_a.*timeout_s=101.*max_timeout_s=100"):
        Session.from_dict(
            _config(
                [{"name": "entry_a", "script": "benchmark.py", "timeout_s": 101}],
                max_timeout_s=100,
            )
        )


def test_session_accepts_timeout_equal_to_max_timeout_s() -> None:
    session = Session.from_dict(
        _config(
            [{"name": "entry_a", "script": "benchmark.py", "timeout_s": 100}],
            max_timeout_s=100,
        )
    )

    assert session.entries[0].timeout_s == 100


@pytest.mark.parametrize("bad_max_timeout_s", [0, -1, True, 1.5])
def test_session_rejects_invalid_max_timeout_s(bad_max_timeout_s: object) -> None:
    with pytest.raises(ValueError, match="Invalid max_timeout_s"):
        Session.from_dict(
            _config(
                [{"name": "entry_a", "script": "benchmark.py"}],
                max_timeout_s=bad_max_timeout_s,
            )
        )


def test_session_applies_max_timeout_s_after_default_timeout_s() -> None:
    with pytest.raises(ValueError, match=r"entry_a.*timeout_s=120.*max_timeout_s=100"):
        Session.from_dict(
            _config(
                [{"name": "entry_a", "script": "benchmark.py"}],
                default_timeout_s=120,
                max_timeout_s=100,
            )
        )


def _generate_job() -> Callable[..., dict]:
    pytest.importorskip("ruamel.yaml")
    from tools.generate_ci_tests import generate_job

    return generate_job


def test_generate_job_rejects_timeout_above_max_timeout_s() -> None:
    generate_job = _generate_job()

    with pytest.raises(ValueError, match=r"entry_a.*timeout_s=101.*max_timeout_s=100"):
        generate_job(
            {"name": "entry_a", "timeout_s": 101},
            "nightly",
            default_timeout_s=120,
            cleanup_timeout_s=60,
            min_timeout_s=600,
            max_timeout_s=100,
        )


def test_generate_job_accepts_timeout_equal_to_max_timeout_s() -> None:
    generate_job = _generate_job()

    job = generate_job(
        {"name": "entry_a", "timeout_s": 100},
        "nightly",
        default_timeout_s=120,
        cleanup_timeout_s=60,
        min_timeout_s=0,
        max_timeout_s=100,
    )

    assert job["variables"]["ENTRY_NAME"] == "entry_a"
    assert job["variables"]["TIME"] == "00:02:40"


def test_generate_job_applies_max_timeout_s_after_default_timeout_s() -> None:
    generate_job = _generate_job()

    with pytest.raises(ValueError, match=r"entry_a.*timeout_s=120.*max_timeout_s=100"):
        generate_job(
            {"name": "entry_a"},
            "nightly",
            default_timeout_s=120,
            cleanup_timeout_s=60,
            min_timeout_s=600,
            max_timeout_s=100,
        )
