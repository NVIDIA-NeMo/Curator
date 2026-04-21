import sys
from types import SimpleNamespace

from nemo_curator.stages.text.utils.text_utils import get_language_name


class _LangResolver:
    def __init__(self, code: str):
        names = {"en": "English", "de": "German"}
        if code not in names:
            raise KeyError(code)
        self.name = names[code]


def test_get_language_name_supports_lang_api(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "iso639", SimpleNamespace(Lang=_LangResolver))
    assert get_language_name("en") == "English"


def test_get_language_name_supports_to_name_api(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "iso639",
        SimpleNamespace(to_name=lambda code: {"en": "English", "de": "German"}[code]),
    )
    assert get_language_name("de") == "German"


def test_get_language_name_falls_back_on_unknown_code(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "iso639",
        SimpleNamespace(to_name=lambda code: (_ for _ in ()).throw(KeyError(code))),
    )
    assert get_language_name("zz") == "zz"
