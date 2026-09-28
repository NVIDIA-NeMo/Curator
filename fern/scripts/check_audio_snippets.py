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

"""Check maintained Fern ``ASRStage`` snippets against the source API."""

from __future__ import annotations

import ast
import html
import re
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ASR_STAGE_SOURCE = REPO_ROOT / "nemo_curator/stages/audio/inference/asr/stage.py"
FERN_PAGES = REPO_ROOT / "fern/versions/main/pages"
PYTHON_FENCE = re.compile(
    r"^[ \t]*```(?:python|py)(?:[ \t][^\n]*)?\n(?P<code>.*?)^[ \t]*```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
ASR_STAGE_TOKEN = re.compile(r"\bASRStage\b")
DOC_REQUIRED_KEYWORDS = {"max_inference_duration_s", "local_bucketing"}


def _field_call(node: ast.expr) -> ast.Call | None:
    if not isinstance(node, ast.Call):
        return None
    name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
    return node if name == "field" else None


def _contains_annotation_name(annotation: ast.expr, name: str) -> bool:
    return any(
        (isinstance(node, ast.Name) and node.id == name) or (isinstance(node, ast.Attribute) and node.attr == name)
        for node in ast.walk(annotation)
    )


def _is_missing_sentinel(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "MISSING") or (
        isinstance(node, ast.Attribute) and node.attr == "MISSING"
    )


def _field_is_in_init(call: ast.Call) -> bool:
    init = next((keyword.value for keyword in call.keywords if keyword.arg == "init"), None)
    return not (isinstance(init, ast.Constant) and init.value is False)


def _field_call_is_required(call: ast.Call) -> bool:
    defaults = [keyword.value for keyword in call.keywords if keyword.arg in {"default", "default_factory"}]
    return not defaults or all(_is_missing_sentinel(default) for default in defaults)


def _asr_stage_fields() -> tuple[set[str], set[str]]:
    tree = ast.parse(ASR_STAGE_SOURCE.read_text(encoding="utf-8"), filename=str(ASR_STAGE_SOURCE))
    stage_class = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ASRStage"),
        None,
    )
    if stage_class is None:
        message = f"Could not find ASRStage in {ASR_STAGE_SOURCE}"
        raise RuntimeError(message)

    required: set[str] = set()
    allowed: set[str] = set()
    for statement in stage_class.body:
        if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
            continue
        if _contains_annotation_name(statement.annotation, "ClassVar") or _contains_annotation_name(
            statement.annotation, "KW_ONLY"
        ):
            continue
        field_call = _field_call(statement.value) if statement.value is not None else None
        if field_call is not None and not _field_is_in_init(field_call):
            continue
        allowed.add(statement.target.id)
        if statement.value is None or (field_call is not None and _field_call_is_required(field_call)):
            required.add(statement.target.id)
    if not required or not allowed:
        message = f"Could not determine ASRStage constructor fields from {ASR_STAGE_SOURCE}"
        raise RuntimeError(message)
    return required, allowed


def _asr_stage_names(tree: ast.Module) -> set[str]:
    names = {"ASRStage"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names if alias.name == "ASRStage")
    return names


def _is_asr_stage_call(node: ast.Call, names: set[str]) -> bool:
    return (isinstance(node.func, ast.Name) and node.func.id in names) or (
        isinstance(node.func, ast.Attribute) and node.func.attr == "ASRStage"
    )


def _literal_number(node: ast.expr) -> float | None:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _validate_call(
    page: Path,
    start_line: int,
    call: ast.Call,
    required_keywords: set[str],
    allowed_keywords: set[str],
) -> list[str]:
    location = f"{page.relative_to(REPO_ROOT)}:{start_line + call.lineno - 1}"
    if call.args or any(keyword.arg is None for keyword in call.keywords):
        return [f"{location}: ASRStage examples must use explicit keyword arguments"]

    errors: list[str] = []
    keywords = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg is not None}
    missing = sorted(required_keywords - keywords.keys())
    if missing:
        errors.append(f"{location}: ASRStage is missing required keywords: {', '.join(missing)}")
    unknown = sorted(keywords.keys() - allowed_keywords)
    if unknown:
        errors.append(f"{location}: ASRStage uses unknown keywords: {', '.join(unknown)}")

    budget = _literal_number(keywords["max_audio_sec_per_actor"]) if "max_audio_sec_per_actor" in keywords else None
    ceiling = _literal_number(keywords["max_inference_duration_s"]) if "max_inference_duration_s" in keywords else None
    if budget is not None and budget <= 0:
        errors.append(f"{location}: max_audio_sec_per_actor must be positive, got {budget}")
    if ceiling is not None and ceiling <= 0:
        errors.append(f"{location}: max_inference_duration_s must be positive, got {ceiling}")
    if budget is not None and ceiling is not None and ceiling > budget:
        errors.append(f"{location}: max_inference_duration_s ({ceiling}) exceeds max_audio_sec_per_actor ({budget})")
    return errors


def main() -> None:
    source_required, source_allowed = _asr_stage_fields()
    required_keywords = source_required | DOC_REQUIRED_KEYWORDS
    errors: list[str] = []
    call_count = 0

    for page in sorted(FERN_PAGES.rglob("*.mdx")):
        contents = page.read_text(encoding="utf-8")
        for fence in PYTHON_FENCE.finditer(contents):
            code = textwrap.dedent(html.unescape(fence.group("code")))
            if ASR_STAGE_TOKEN.search(code) is None:
                continue
            start_line = contents.count("\n", 0, fence.start("code")) + 1
            try:
                tree = ast.parse(code, filename=str(page))
            except SyntaxError as exc:
                errors.append(f"{page.relative_to(REPO_ROOT)}:{start_line + (exc.lineno or 1) - 1}: {exc.msg}")
                continue

            asr_stage_names = _asr_stage_names(tree)
            for call in (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and _is_asr_stage_call(node, asr_stage_names)
            ):
                call_count += 1
                errors.extend(_validate_call(page, start_line, call, required_keywords, source_allowed))

    if call_count == 0:
        errors.append(f"No ASRStage calls found below {FERN_PAGES.relative_to(REPO_ROOT)}")
    if errors:
        raise SystemExit("Fern audio snippet validation failed:\n" + "\n".join(f"- {error}" for error in errors))

    fields = ", ".join(sorted(required_keywords))
    print(f"Validated {call_count} Fern ASRStage calls against required keywords: {fields}")


if __name__ == "__main__":
    main()
