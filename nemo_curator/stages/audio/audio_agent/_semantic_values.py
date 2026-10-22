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

"""JSON-safe configured recipe values and identity summaries."""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any


def _json_safe(value: Any) -> Any:  # noqa: ANN401
    """Recursively preserve JSON values and stringify an unexpected leaf."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value, key=repr)]
    return str(value)


def _recipe_stage_ref(recipe: Any, index: int, stage: Any) -> str:  # noqa: ANN401
    """Resolve the authored recipe ref without requiring a concrete Recipe type."""
    entries: Any = getattr(recipe, "stages", None)
    if entries is None and isinstance(recipe, Mapping):
        entries = recipe.get("stages")
    if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)) and index < len(entries):
        entry = entries[index]
        ref = getattr(entry, "ref", None)
        if ref is None and isinstance(entry, Mapping):
            ref = entry.get("ref")
        if ref:
            return str(ref)
    return type(stage).__name__


def _recipe_stage_params(recipe: Any, index: int) -> dict[str, Any]:  # noqa: ANN401
    """Return the authored params for one recipe entry, if available."""
    entries: Any = getattr(recipe, "stages", None)
    if entries is None and isinstance(recipe, Mapping):
        entries = recipe.get("stages")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or index >= len(entries):
        return {}
    entry = entries[index]
    params = getattr(entry, "params", None)
    if params is None and isinstance(entry, Mapping):
        params = entry.get("params")
    return dict(params) if isinstance(params, Mapping) else {}


def _jsonable(value: Any) -> Any:  # noqa: ANN401
    """Bound configured values to JSON-safe data without executing anything."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        with contextlib.suppress(Exception):
            return _jsonable(value.to_dict())
    return f"<{type(value).__name__}>"


def _configured_params(
    stage: Any,  # noqa: ANN401
    contract: Any,  # noqa: ANN401
    authored: dict[str, Any],
    *,
    implicit_source: str = "default",
) -> dict[str, dict[str, Any]]:
    """Effective parameter evidence, preserving whether a value was explicit."""
    from nemo_curator.stages.audio.audio_agent import _safety

    configured: dict[str, dict[str, Any]] = {}
    missing = object()
    specs = {str(getattr(param, "name", "")): param for param in (getattr(contract, "params", ()) or ())}
    for name, spec in specs.items():
        if not name:
            continue
        if name in authored:
            value = authored[name]
            source = "recipe"
        else:
            try:
                value = getattr(stage, name)
            except Exception:  # noqa: BLE001 - advisory inspection stays best-effort
                value = missing
            if callable(value):
                callable_name = getattr(value, "__name__", None)
                choices = getattr(spec, "choices", None) or ()
                value = (
                    callable_name
                    if isinstance(callable_name, str)
                    and (callable_name in choices or str(getattr(spec, "type", "")) == "str")
                    else missing
                )
            if value is missing:
                value = getattr(spec, "default", None)
            source = implicit_source
        configured[name] = {"value": _jsonable(value), "source": source}
    # Framework knobs may not appear in the stage constructor contract but still
    # affect the authored plan. Include them as explicit evidence.
    for name, value in authored.items():
        configured.setdefault(str(name), {"value": _jsonable(value), "source": "recipe"})
    return _safety.redact(configured)


def _best_effort_instance_params(
    stage: Any,  # noqa: ANN401
    *,
    source: str,
) -> dict[str, dict[str, Any]]:
    """Configured values for a leaf whose AgentReady contract is unavailable."""
    from nemo_curator.stages.audio.audio_agent import _safety

    names: list[str] = []
    if dataclasses.is_dataclass(stage) and not isinstance(stage, type):
        names = [field.name for field in dataclasses.fields(stage) if field.init and not field.name.startswith("_")]
    else:
        with contextlib.suppress(TypeError, ValueError):
            import inspect

            names = [
                name
                for name, param in inspect.signature(type(stage).__init__).parameters.items()
                if name != "self"
                and not name.startswith("_")
                and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
            ]
    excluded = {"name", "resources", "batch_size", "runtime_env", "num_workers"}
    configured: dict[str, dict[str, Any]] = {}
    for name in names:
        if name in excluded:
            continue
        with contextlib.suppress(Exception):
            value = getattr(stage, name)
            if not callable(value):
                configured[name] = {"value": _jsonable(value), "source": source}
    return _safety.redact(configured)


def _recipe_value(recipe: Any, name: str, default: Any = None) -> Any:  # noqa: ANN401
    value = getattr(recipe, name, default)
    if isinstance(recipe, Mapping):
        value = recipe.get(name, value)
    return value


def _recipe_identity(recipe: Any, stage_count: int) -> dict[str, Any]:  # noqa: ANN401
    from nemo_curator.stages.audio.audio_agent import _safety

    if recipe is None:
        return {"stage_count": stage_count}
    authored_config_hash = _recipe_value(recipe, "config_hash")
    config_hash: str | None = None
    hash_source: str | None = None
    compute_hash = getattr(recipe, "compute_hash", None)
    if callable(compute_hash):
        with contextlib.suppress(Exception):
            config_hash = str(compute_hash())
            hash_source = "computed_canonical_recipe"
    elif isinstance(recipe, Mapping):
        # Public callers may pass the same JSON-safe recipe mapping accepted by
        # validate. Compute its portable canonical hash without mutating it.
        with contextlib.suppress(Exception):
            from nemo_curator.stages.audio.audio_agent.recipe import Recipe

            config_hash = Recipe.from_dict(dict(recipe)).compute_hash()
            hash_source = "computed_canonical_recipe"
    if config_hash is None and authored_config_hash:
        # This fallback is useful to non-Recipe SDK callers, but is explicitly
        # labeled unverified rather than silently treating a supplied value as
        # the canonical identity.
        config_hash = str(authored_config_hash)
        hash_source = "provided_unverified"
    recipe_id = _recipe_value(recipe, "recipe_id")
    out: dict[str, Any] = {"stage_count": stage_count}
    if recipe_id:
        out["recipe_id"] = str(recipe_id)
    if config_hash:
        out["config_hash"] = config_hash
        out["config_hash_source"] = hash_source
    if authored_config_hash and config_hash and str(authored_config_hash) != config_hash:
        out["authored_config_hash_mismatch"] = True
    for name in ("preset", "rationale", "acceptance_criteria", "config_strategy"):
        value = _recipe_value(recipe, name)
        if value not in (None, "", [], {}):
            out[name] = _jsonable(value)
    return _safety.redact(out)


def _data_profile_summary(data_profile: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return only bounded, critique-relevant profile facts (never examples/content)."""
    if not isinstance(data_profile, Mapping):
        return None
    from nemo_curator.stages.audio.audio_agent import _safety

    fields = (
        "kind",
        "num_files",
        "sample_rates",
        "channels",
        "total_duration_sec",
        "mean_duration_sec",
        "codecs",
        "has_transcripts",
        "manifest_keys",
        "unreadable",
        "source_errors",
        "notes",
        "fingerprint_tier",
    )
    summary = {name: _jsonable(data_profile[name]) for name in fields if name in data_profile}
    return _safety.redact(summary)
