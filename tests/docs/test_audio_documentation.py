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

import ast
import html
import re
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
FERN_PAGES = REPO_ROOT / "fern" / "versions" / "main" / "pages"
AUDIO_TUTORIALS = REPO_ROOT / "tutorials" / "audio"
PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)
REMOVED_AUDIO_APIS = (
    "InferenceAsrNemoStage",
    "nemo_curator.stages.audio.inference.asr_nemo",
    "nemo_curator.stages.audio.metrics.get_wer",
    "get_cer(",
    "get_wordrate(",
    "get_charrate(",
)


def _audio_docs() -> list[Path]:
    paths = list((FERN_PAGES / "curate-audio").rglob("*.mdx"))
    paths.extend((FERN_PAGES / "about" / "concepts" / "audio").rglob("*.mdx"))
    paths.append(FERN_PAGES / "get-started" / "audio.mdx")
    paths.extend(AUDIO_TUTORIALS.rglob("*.md"))
    return sorted(paths)


def _python_fences(path: Path) -> Iterator[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    for match in PYTHON_FENCE.finditer(text):
        line = text[: match.start()].count("\n") + 2
        yield line, html.unescape(match.group(1))


def _validate_asr_call(keywords: dict[str, ast.expr], location: str) -> list[str]:
    failures = []
    missing = sorted({"adapter_target", "model_id"} - keywords.keys())
    legacy = sorted({"filepath_key", "model_name"} & keywords.keys())
    if missing:
        failures.append(f"{location}: ASRStage missing {', '.join(missing)}")
    if legacy:
        failures.append(f"{location}: ASRStage uses removed keyword(s) {', '.join(legacy)}")
    return failures


def _validate_fleurs_call(keywords: dict[str, ast.expr], location: str) -> list[str]:
    failures = []
    split = keywords.get("split")
    if isinstance(split, ast.Constant) and split.value not in {"train", "dev", "test"}:
        failures.append(f"{location}: unsupported FLEURS split {split.value!r}")
    raw_data_dir = keywords.get("raw_data_dir")
    if isinstance(raw_data_dir, ast.Constant) and str(raw_data_dir.value).startswith("~"):
        failures.append(f"{location}: raw_data_dir does not expand '~'")
    return failures


def test_audio_docs_do_not_reference_removed_apis() -> None:
    failures = []
    for path in _audio_docs():
        text = path.read_text(encoding="utf-8")
        for removed_api in REMOVED_AUDIO_APIS:
            if removed_api in text:
                failures.append(f"{path.relative_to(REPO_ROOT)} references {removed_api!r}")
    assert not failures, "\n".join(failures)


def test_audio_python_fences_parse() -> None:
    failures = []
    for path in _audio_docs():
        for line, code in _python_fences(path):
            try:
                ast.parse(code)
            except SyntaxError as exc:
                failures.append(f"{path.relative_to(REPO_ROOT)}:{line + (exc.lineno or 1) - 1}: {exc.msg}")
    assert not failures, "\n".join(failures)


def test_audio_stage_snippets_use_current_runtime_contracts() -> None:
    failures = []
    for path in _audio_docs():
        for line, code in _python_fences(path):
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg is not None}
                location = f"{path.relative_to(REPO_ROOT)}:{line + node.lineno - 1}"
                if node.func.id == "ASRStage":
                    failures.extend(_validate_asr_call(keywords, location))
                elif node.func.id == "CreateInitialManifestFleursStage":
                    failures.extend(_validate_fleurs_call(keywords, location))
    assert not failures, "\n".join(failures)


def test_tagging_usage_anchors_the_bundled_manifest_to_repo_root() -> None:
    source = (AUDIO_TUTORIALS / "tagging" / "main.py").read_text(encoding="utf-8")
    usage = ast.get_docstring(ast.parse(source)) or ""
    readme = (AUDIO_TUTORIALS / "tagging" / "README.md").read_text(encoding="utf-8")

    assert 'input_manifest="${PWD}/tests/fixtures/audio/tagging/sample_input.jsonl"' in usage
    assert 'input_manifest="${PWD}/tests/fixtures/audio/tagging/sample_input.jsonl"' in readme
    assert "input_manifest=tests/fixtures/audio/tagging/sample_input.jsonl" not in usage
    assert "input_manifest=tests/fixtures/audio/tagging/sample_input.jsonl" not in readme
