# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_build_engine_module() -> ModuleType:
    script = Path(__file__).parents[4] / "scripts" / "audio" / "indic_canary_trtllm_conversion" / "build_engine.py"
    spec = importlib.util.spec_from_file_location("indic_canary_build_engine", script)
    if spec is None or spec.loader is None:
        msg = f"Could not load build helper from {script}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_engine = _load_build_engine_module()


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(30.0, (3001, 246)), (40.0, (4001, 374))],
)
def test_derive_engine_lengths_matches_known_engine_profiles(
    seconds: float,
    expected: tuple[int, int],
) -> None:
    assert build_engine._derive_engine_lengths(seconds, 10) == expected


def test_parse_args_requires_an_explicit_indic_canary_source() -> None:
    with pytest.raises(SystemExit):
        build_engine.parse_args(["--engine_dir", "/models/engine"])


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--max_batch_size", "0"),
        ("--max_beam_width", "0"),
        ("--max_prompt_tokens", "0"),
        ("--max_audio_seconds", "nan"),
        ("--max_audio_seconds", "0"),
    ],
)
def test_parse_args_rejects_invalid_engine_limits(flag: str, value: str) -> None:
    with pytest.raises(SystemExit):
        build_engine.parse_args(
            [
                "--model_name",
                "organization/indic-canary",
                "--engine_dir",
                "/models/engine",
                flag,
                value,
            ]
        )


def test_validate_engine_requires_all_runtime_artifacts_to_be_files(tmp_path: Path) -> None:
    for relative_path in build_engine._REQUIRED_ARTIFACTS:
        artifact = tmp_path / relative_path
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.touch()

    build_engine._validate_engine(tmp_path)

    broken = tmp_path / build_engine._REQUIRED_ARTIFACTS[0]
    broken.unlink()
    broken.mkdir()
    with pytest.raises(FileNotFoundError, match=r"encoder/encoder\.plan"):
        build_engine._validate_engine(tmp_path)


def test_convert_command_uses_the_explicit_model_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = build_engine.parse_args(
        [
            "--model_name",
            "organization/indic-canary",
            "--engine_dir",
            str(tmp_path / "engine"),
        ]
    )
    captured: list[list[str]] = []
    monkeypatch.setattr(build_engine, "_run", lambda cmd, **_kwargs: captured.append(cmd))

    build_engine._convert_checkpoint(args, tmp_path / "checkpoint", tmp_path / "engine", {})

    assert captured[0][-3:] == ["--model_name", "organization/indic-canary", str(tmp_path / "engine")]
