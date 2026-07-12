# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""CPU-only tests for the lazy-import mechanism in stages/text/classifiers/__init__.py.

All tests run without a GPU.  Heavy deps (torch, transformers) are mocked so
that importing the package attributes does not trigger real model loading.
"""

import importlib
import sys
import types
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PACKAGE = "nemo_curator.stages.text.classifiers"

# Mapping copied verbatim from __init__.py — changing it here should be
# accompanied by a change there.
_LAZY: dict[str, str] = {
    "AegisClassifier": ".aegis",
    "InstructionDataGuardClassifier": ".aegis",
    "ContentTypeClassifier": ".content_type",
    "DomainClassifier": ".domain",
    "MultilingualDomainClassifier": ".domain",
    "FineWebEduClassifier": ".fineweb_edu",
    "FineWebMixtralEduClassifier": ".fineweb_edu",
    "FineWebNemotronEduClassifier": ".fineweb_edu",
    "PromptTaskComplexityClassifier": ".prompt_task_complexity",
    "QualityClassifier": ".quality",
}


def _make_fake_module(class_name: str) -> types.ModuleType:
    """Return a minimal module stub that exposes *class_name* as a sentinel."""
    mod = types.ModuleType(class_name + "_module")
    sentinel = type(class_name, (), {})  # a unique class object per name
    setattr(mod, class_name, sentinel)
    return mod


# ---------------------------------------------------------------------------
# RAPIDS_NO_INITIALIZE
# ---------------------------------------------------------------------------


class TestRapidsNoInitialize:
    def test_env_var_set_on_import(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Remove the env var so we can observe the package setting it.
        monkeypatch.delenv("RAPIDS_NO_INITIALIZE", raising=False)

        # Re-import by removing the cached module and re-importing.
        saved = sys.modules.pop(_PACKAGE, None)
        try:
            importlib.import_module(_PACKAGE)
            import os

            assert os.environ.get("RAPIDS_NO_INITIALIZE") == "1"
        finally:
            # Restore original cached module to avoid polluting other tests.
            if saved is not None:
                sys.modules[_PACKAGE] = saved
            else:
                sys.modules.pop(_PACKAGE, None)


# ---------------------------------------------------------------------------
# __all__ completeness
# ---------------------------------------------------------------------------


class TestDunderAll:
    def test_all_contains_every_lazy_name(self):
        pkg = importlib.import_module(_PACKAGE)
        assert set(pkg.__all__) == set(_LAZY.keys())

    def test_all_has_no_extra_names(self):
        pkg = importlib.import_module(_PACKAGE)
        assert len(pkg.__all__) == len(_LAZY)


# ---------------------------------------------------------------------------
# __getattr__ — known names resolve to the correct class
# ---------------------------------------------------------------------------


class TestGetattr:
    def _fetch_with_mock(self, name: str) -> type:
        """Import *name* from the package with importlib.import_module mocked."""
        relative_submodule = _LAZY[name]

        fake_mod = _make_fake_module(name)

        pkg = importlib.import_module(_PACKAGE)

        with patch("importlib.import_module", return_value=fake_mod) as mock_import:
            result = pkg.__getattr__(name)

        # Verify that import_module was called with the right arguments.
        mock_import.assert_called_once_with(relative_submodule, package=_PACKAGE)
        return result

    def test_known_name_returns_class(self):
        name = "DomainClassifier"
        cls = self._fetch_with_mock(name)
        # The sentinel class carries the same name.
        assert cls.__name__ == name

    def test_all_lazy_names_resolve(self):
        for name in _LAZY:
            cls = self._fetch_with_mock(name)
            assert cls.__name__ == name, f"{name} did not resolve correctly"

    def test_unknown_name_raises_attribute_error(self):
        pkg = importlib.import_module(_PACKAGE)
        with pytest.raises(AttributeError, match="has no attribute"):
            pkg.__getattr__("NonExistentClassifier")

    def test_attribute_error_message_contains_module_name(self):
        pkg = importlib.import_module(_PACKAGE)
        with pytest.raises(AttributeError) as exc_info:
            pkg.__getattr__("NotReal")
        assert _PACKAGE in str(exc_info.value)
        assert "NotReal" in str(exc_info.value)


# ---------------------------------------------------------------------------
# No heavy imports at package-import time
# ---------------------------------------------------------------------------


class TestNoDeferredImportsAtParseTime:
    """Verify that importing the classifiers package does not eagerly pull in
    torch or transformers.  These deps are expensive and GPU-only; they must
    stay out of the module-level namespace."""

    def _reimport_package(self) -> None:
        saved = sys.modules.pop(_PACKAGE, None)
        try:
            importlib.import_module(_PACKAGE)
        finally:
            if saved is not None:
                sys.modules[_PACKAGE] = saved
            else:
                sys.modules.pop(_PACKAGE, None)

    def test_torch_not_imported_eagerly(self):
        torch_before = "torch" in sys.modules
        self._reimport_package()
        if not torch_before:
            assert "torch" not in sys.modules, "torch was imported at package parse time"

    def test_transformers_not_imported_eagerly(self):
        transformers_before = "transformers" in sys.modules
        self._reimport_package()
        if not transformers_before:
            assert "transformers" not in sys.modules, "transformers was imported at package parse time"
