import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_curator.stages.text.experimental.translation.utils.prompt_loader import (
    load_prompt_template,
)
from nemo_curator.stages.text.utils.text_utils import get_language_name


class _LangResolver:
    def __init__(self, code: str):
        names = {"en": "English", "de": "German"}
        if code not in names:
            raise KeyError(code)
        self.name = names[code]


def test_get_language_name_supports_lang_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "iso639", SimpleNamespace(Lang=_LangResolver))
    assert get_language_name("en") == "English"


def test_get_language_name_supports_to_name_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "iso639",
        SimpleNamespace(to_name=lambda code: {"en": "English", "de": "German"}[code]),
    )
    assert get_language_name("de") == "German"


def test_get_language_name_falls_back_on_unknown_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "iso639",
        SimpleNamespace(to_name=lambda code: (_ for _ in ()).throw(KeyError(code))),
    )
    assert get_language_name("zz") == "zz"


def test_load_prompt_template_supports_absolute_path(tmp_path: Path) -> None:
    prompt_path = tmp_path / "custom_translate.yaml"
    prompt_path.write_text("system: custom system\nuser: custom user {src}\n", encoding="utf-8")

    system_prompt, user_template = load_prompt_template(prompt_path)

    assert system_prompt == "custom system"
    assert user_template == "custom user {src}"


def test_load_prompt_template_rejects_missing_required_keys(tmp_path: Path) -> None:
    prompt_path = tmp_path / "bad_prompt.yaml"
    prompt_path.write_text("system: custom system\n", encoding="utf-8")

    with pytest.raises(KeyError, match="missing required keys"):
        load_prompt_template(prompt_path)
