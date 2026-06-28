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
# ruff: noqa: ANN401, ARG001, BLE001, PLR0913, PLR2004, S110, TRY004

"""Generic pipeline expansion for payload handle lifecycles."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.stages.base import ProcessingStage


@dataclass(frozen=True)
class PayloadBindingSpec:
    source_key: str
    ref_key: str
    waveform_key: str
    sample_rate_key: str
    num_samples_key: str
    duration_key: str
    materialize_stage_name: str


def expand_payload_lifecycle_stages(
    stages: list[ProcessingStage],
    config: Any,
) -> list[ProcessingStage]:
    """Insert payload materialize/release helpers around logical stages.

    The lifecycle config is pipeline-level and backend-neutral. It keeps compute
    stages visible to executors while adding only the mechanical stages that own
    payload I/O and cleanup.
    """

    payload_cfg = _config_section(config, "payload_lifecycle")
    if not bool(payload_cfg.get("enabled", False)):
        return stages

    payload_release_stage_cls = _payload_release_stage_class()
    materialize_idx, release_idx, consumers = _payload_lifecycle_positions(stages, payload_cfg)
    run_id = _pipeline_run_id(config)

    reader = _last_manifest_reader(stages[: materialize_idx + 1])
    payload_specs = _payload_binding_specs(payload_cfg, stages=stages, consumers=consumers, reader=reader)
    _configure_planned_source_segment_inputs(reader, payload_cfg, payload_specs, config)
    _validate_payload_consumers(consumers, payload_specs)
    _validate_single_segment_planner_owner(
        reader,
        consumers,
        config=config,
    )

    materializers = [
        _build_payload_materializer(reader, spec, payload_cfg, config, run_id=run_id) for spec in payload_specs
    ]
    primary_spec = payload_specs[0]
    release = payload_release_stage_cls(
        name=str(payload_cfg.get("release_stage_name", "payload_release")),
        payload_ref_key=primary_spec.ref_key,
        waveform_key=primary_spec.waveform_key,
    )

    assembler = _post_release_payload_lifecycle_stage(config, reader, consumers, primary_spec, run_id=run_id)
    execution_source = _payload_lifecycle_source_stage(reader)
    extended_metrics = any(bool(getattr(stage, "extended_performance_metrics", False)) for stage in stages)
    for helper in [*materializers, release, *([assembler] if assembler is not None else [])]:
        helper.extended_performance_metrics = extended_metrics

    # Keep lifecycle bookkeeping out of unrelated pipelines. Only stages that
    # can actually receive refs pay the recursive ref-scan cost, and only the
    # global segmented path preserves terminal rows for downstream assembly.
    for stage in stages[materialize_idx + 1 : release_idx + 1]:
        stage._curator_tracks_payload_refs = True
    if assembler is not None:
        for stage in [*materializers, *stages[materialize_idx + 1 : release_idx + 1]]:
            stage._curator_preserves_terminal_tasks = True

    expanded: list[ProcessingStage] = []
    for idx, stage in enumerate(stages):
        expanded.append(execution_source if stage is reader else stage)
        if idx == materialize_idx:
            expanded.extend(materializers)
        if idx == release_idx:
            expanded.append(release)
            if assembler is not None:
                expanded.append(assembler)
    logger.info(
        "Expanded logical graph into payload lifecycle execution graph: {}",
        " -> ".join(stage.name for stage in expanded),
    )
    return expanded


def _payload_lifecycle_source_stage(reader: ProcessingStage | None) -> ProcessingStage | None:
    """Let a modality-owned planner replace its logical reader at execution time."""
    if reader is None:
        return None
    builder = getattr(reader, "build_payload_lifecycle_source_stage", None)
    if not callable(builder):
        return reader
    source_stage = builder()
    if source_stage is None:
        msg = f"{type(reader).__name__}.build_payload_lifecycle_source_stage() returned None"
        raise TypeError(msg)
    return source_stage


def _payload_lifecycle_positions(
    stages: list[ProcessingStage], payload_cfg: dict[str, Any]
) -> tuple[int, int, list[ProcessingStage]]:
    """Validate selectors and resolve lifecycle stage positions."""
    helpers = [stage.name for stage in stages if bool(getattr(stage, "_curator_pipeline_helper_stage", False))]
    if helpers:
        msg = (
            "Payload lifecycle configs must declare logical stages only. Do not list payload "
            "materialization, payload release, or other pipeline helper stages explicitly. "
            f"Remove implementation helper stage(s): {helpers}"
        )
        raise ValueError(msg)

    materialize_after = _single_selector(
        payload_cfg.get("materialize_after"), key="payload_lifecycle.materialize_after"
    )
    release_after = _single_selector(payload_cfg.get("release_after"), key="payload_lifecycle.release_after")
    consumer_selectors = _normalise_string_list(payload_cfg.get("consumers"), key="payload_lifecycle.consumers")

    materialize_idx = _find_stage_index(stages, materialize_after, key="payload_lifecycle.materialize_after")
    release_idx = _find_stage_index(stages, release_after, key="payload_lifecycle.release_after")
    if release_idx <= materialize_idx:
        msg = "payload_lifecycle.release_after must come after payload_lifecycle.materialize_after"
        raise ValueError(msg)
    consumer_indices = [
        _find_stage_index(stages, selector, key="payload_lifecycle.consumers") for selector in consumer_selectors
    ]
    if any(idx <= materialize_idx or idx > release_idx for idx in consumer_indices):
        msg = (
            "payload_lifecycle.consumers must appear after materialize_after and no later than release_after; "
            f"got consumer indices {consumer_indices}, materialize_after={materialize_idx}, release_after={release_idx}"
        )
        raise ValueError(msg)
    return materialize_idx, release_idx, [stages[idx] for idx in consumer_indices]


def _payload_binding_specs(
    payload_cfg: dict[str, Any],
    *,
    stages: list[ProcessingStage],
    consumers: list[ProcessingStage],
    reader: ProcessingStage | None,
) -> list[PayloadBindingSpec]:
    payloads = payload_cfg.get("payloads")
    if payloads:
        payload_entries = _as_container(payloads)
        if not isinstance(payload_entries, list):
            msg = "payload_lifecycle.payloads must be a list of mappings"
            raise TypeError(msg)
        specs = [
            _payload_spec_from_mapping(dict(entry), payload_cfg=payload_cfg, index=idx)
            for idx, entry in enumerate(payload_entries)
        ]
    else:
        payload_keys = _normalise_string_list(
            payload_cfg.get("payload_keys", ["audio_filepath"]), key="payload_lifecycle.payload_keys"
        )
        specs = [
            _payload_spec_from_legacy_key(
                source_key=source_key,
                payload_cfg=payload_cfg,
                stages=stages,
                consumers=consumers,
                reader=reader,
                index=idx,
                total=len(payload_keys),
            )
            for idx, source_key in enumerate(payload_keys)
        ]
    _validate_unique([spec.source_key for spec in specs], "payload source key")
    _validate_unique([spec.ref_key for spec in specs], "payload ref key")
    _validate_unique([spec.waveform_key for spec in specs], "payload waveform key")
    return specs


def _payload_spec_from_mapping(
    entry: dict[str, Any], *, payload_cfg: dict[str, Any], index: int
) -> PayloadBindingSpec:
    source_key = str(
        entry.get("source_key") or entry.get("payload_key") or entry.get("audio_filepath_key") or ""
    ).strip()
    if not source_key:
        msg = f"payload_lifecycle.payloads[{index}] requires source_key"
        raise ValueError(msg)
    return PayloadBindingSpec(
        source_key=source_key,
        ref_key=str(entry.get("ref_key") or _derived_key(source_key, "ref")),
        waveform_key=str(entry.get("waveform_key") or _derived_key(source_key, "waveform")),
        sample_rate_key=str(entry.get("sample_rate_key") or _derived_key(source_key, "sample_rate")),
        num_samples_key=str(entry.get("num_samples_key") or _derived_key(source_key, "num_samples")),
        duration_key=str(entry.get("duration_key") or payload_cfg.get("duration_key", "duration")),
        materialize_stage_name=str(
            entry.get("materialize_stage_name") or _materialize_stage_name(payload_cfg, index, source_key)
        ),
    )


def _payload_spec_from_legacy_key(
    *,
    source_key: str,
    payload_cfg: dict[str, Any],
    stages: list[ProcessingStage],
    consumers: list[ProcessingStage],
    reader: ProcessingStage | None,
    index: int,
    total: int,
) -> PayloadBindingSpec:
    if total == 1:
        ref_key = str(
            payload_cfg.get("ref_key") or _consumer_payload_key(consumers, "waveform_ref_key", "waveform_ref")
        )
        waveform_key = str(
            payload_cfg.get("waveform_key") or _consumer_payload_key(consumers, "waveform_key", "waveform")
        )
        sample_rate_key = str(
            payload_cfg.get("sample_rate_key") or _consumer_payload_key(consumers, "sample_rate_key", "sample_rate")
        )
        num_samples_key = str(
            payload_cfg.get("num_samples_key") or _first_attr(stages, "num_samples_key", "num_samples")
        )
        duration_key = str(
            payload_cfg.get("duration_key")
            or _first_attr(stages, "duration_key", getattr(reader, "duration_key", "duration"))
        )
    else:
        ref_key = _list_or_derived(payload_cfg, "ref_keys", index, source_key, "ref")
        waveform_key = _list_or_derived(payload_cfg, "waveform_keys", index, source_key, "waveform")
        sample_rate_key = _list_or_derived(payload_cfg, "sample_rate_keys", index, source_key, "sample_rate")
        num_samples_key = _list_or_derived(payload_cfg, "num_samples_keys", index, source_key, "num_samples")
        duration_key = _list_or_default(
            payload_cfg, "duration_keys", index, str(_first_attr(stages, "duration_key", "duration"))
        )
    return PayloadBindingSpec(
        source_key=source_key,
        ref_key=ref_key,
        waveform_key=waveform_key,
        sample_rate_key=sample_rate_key,
        num_samples_key=num_samples_key,
        duration_key=duration_key,
        materialize_stage_name=_materialize_stage_name(payload_cfg, index, source_key, total=total),
    )


def _validate_payload_consumers(consumers: list[ProcessingStage], payload_specs: list[PayloadBindingSpec]) -> None:
    by_ref = {spec.ref_key: spec for spec in payload_specs}
    for stage in consumers:
        if not hasattr(stage, "resolve_payload_refs_for_batch"):
            msg = (
                f"Payload consumer {stage.name!r} is not payload-aware. Stages that consume payload refs must "
                "implement PayloadAwareStageMixin or an equivalent resolve_payload_refs_for_batch() contract."
            )
            raise TypeError(msg)
        bindings = _stage_payload_bindings(stage)
        if not bindings:
            msg = f"Payload consumer {stage.name!r} does not declare any payload ref bindings"
            raise ValueError(msg)
        for binding in bindings:
            ref_key = str(binding["ref_key"])
            if ref_key not in by_ref:
                msg = (
                    f"Payload consumer {stage.name!r} declares ref_key={ref_key!r}, but payload_lifecycle "
                    f"materializes only {sorted(by_ref)}"
                )
                raise ValueError(msg)
            expected_waveform_key = by_ref[ref_key].waveform_key
            waveform_key = str(binding.get("waveform_key") or "")
            if waveform_key != expected_waveform_key:
                msg = (
                    f"Payload consumer {stage.name!r} must declare waveform_key={expected_waveform_key!r} "
                    f"for ref_key={ref_key!r}; got {waveform_key!r}"
                )
                raise ValueError(msg)


def _stage_payload_bindings(stage: ProcessingStage) -> list[dict[str, str]]:
    bindings = getattr(stage, "payload_bindings", None)
    if callable(bindings):
        bindings = bindings()
    if bindings:
        result = []
        for item in bindings:
            if not isinstance(item, dict):
                msg = f"{stage.name}.payload_bindings entries must be mappings"
                raise TypeError(msg)
            ref_key = str(item.get("ref_key") or "").strip()
            waveform_key = str(item.get("waveform_key") or "").strip()
            if ref_key and waveform_key:
                result.append(
                    {**{str(k): str(v) for k, v in item.items()}, "ref_key": ref_key, "waveform_key": waveform_key}
                )
        return result
    ref_key = getattr(stage, "waveform_ref_key", None)
    waveform_key = getattr(stage, "waveform_key", None)
    if ref_key and waveform_key:
        return [
            {
                "ref_key": str(ref_key),
                "waveform_key": str(waveform_key),
                "sample_rate_key": str(getattr(stage, "sample_rate_key", "sample_rate")),
                "num_samples_key": str(getattr(stage, "num_samples_key", "num_samples")),
            }
        ]
    return []


def _configure_planned_source_segment_inputs(
    reader: ProcessingStage | None,
    payload_cfg: dict[str, Any],
    payload_specs: list[PayloadBindingSpec],
    config: Any,
) -> None:
    if reader is None or not bool(getattr(reader, "enable_global_bucketing", False)):
        return
    scheduler_cfg = _config_section(config, "global_audio_scheduler")
    configured = scheduler_cfg.get("segment_input_keys", payload_cfg.get("segment_input_keys"))
    segment_input_keys: list[str] = []
    if configured is not None:
        segment_input_keys.extend(_normalise_string_list(configured, key="global_audio_scheduler.segment_input_keys"))
    segment_input_keys.extend(spec.source_key for spec in payload_specs)
    reader.segment_input_keys = _dedupe_strings(segment_input_keys)
    reader.run_id = _pipeline_run_id(config)
    if "parent_store_actor_name_prefix" in scheduler_cfg:
        reader.parent_store_actor_name_prefix = str(scheduler_cfg["parent_store_actor_name_prefix"])


def _validate_single_segment_planner_owner(
    reader: ProcessingStage | None,
    consumers: list[ProcessingStage],
    *,
    config: Any,
) -> None:
    if reader is None or not bool(getattr(reader, "enable_global_bucketing", False)):
        return
    owner_stage = _single_selector(getattr(reader, "owner_stage", None), key="global_audio_scheduler.owner_stage")
    matching_consumers = [stage for stage in consumers if owner_stage in _stage_match_idents(stage)]
    if not matching_consumers:
        available = sorted({ident for stage in consumers for ident in _stage_match_idents(stage)})
        msg = (
            "global_audio_scheduler.owner_stage must select exactly one stage listed in "
            "payload_lifecycle.consumers. Global bucketing has a single planning owner; "
            f"{owner_stage!r} was not found in payload consumers {available}."
        )
        raise ValueError(msg)
    if len(matching_consumers) > 1:
        names = [stage.name for stage in matching_consumers]
        msg = f"global_audio_scheduler.owner_stage must select exactly one payload consumer; matched {names}"
        raise ValueError(msg)
    _validate_planner_owner_has_largest_model_window(reader=reader, owner=matching_consumers[0], consumers=consumers)
    reader.owner_stage = owner_stage


def _validate_planner_owner_has_largest_model_window(
    *,
    reader: ProcessingStage,
    owner: ProcessingStage,
    consumers: list[ProcessingStage],
) -> None:
    owner_max_s = _required_positive_seconds(owner, "max_inference_duration_s")
    consumer_max_s = [
        (stage.name, _required_positive_seconds(stage, "max_inference_duration_s")) for stage in consumers
    ]
    larger_consumers = [(name, max_s) for name, max_s in consumer_max_s if max_s > owner_max_s]
    if larger_consumers:
        details = ", ".join(f"{name}={value:g}s" for name, value in larger_consumers)
        msg = (
            "global_audio_scheduler.owner_stage must select the payload consumer with the largest "
            "max_inference_duration_s because the source planner emits one segment plan. "
            f"Selected owner {owner.name!r} has max_inference_duration_s={owner_max_s:g}s, "
            f"but larger consumer(s) exist: {details}."
        )
        raise ValueError(msg)

    reader_max_s = _required_positive_seconds(reader, "max_inference_duration_s")
    if abs(reader_max_s - owner_max_s) > 1e-6:
        msg = (
            "ManifestReader(enable_global_bucketing=True).max_inference_duration_s must match the "
            "selected owner stage's max_inference_duration_s. "
            f"Reader has {reader_max_s:g}s, owner {owner.name!r} has {owner_max_s:g}s."
        )
        raise ValueError(msg)


def _required_positive_seconds(stage: ProcessingStage, attr: str) -> float:
    value = getattr(stage, attr, None)
    if value is None:
        msg = f"Global bucketing requires stage {stage.name!r} to define positive {attr}"
        raise ValueError(msg)
    return _positive_seconds(value, label=f"{stage.name}.{attr}")


def _optional_positive_seconds(stage: ProcessingStage, attr: str) -> float | None:
    value = getattr(stage, attr, None)
    if value is None:
        return None
    return _positive_seconds(value, label=f"{stage.name}.{attr}")


def _positive_seconds(value: Any, *, label: str) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        msg = f"{label} must be a positive number of seconds, got {value!r}"
        raise TypeError(msg) from exc
    if seconds <= 0:
        msg = f"{label} must be > 0 seconds, got {seconds:g}"
        raise ValueError(msg)
    return seconds


def _build_payload_materializer(
    reader: ProcessingStage | None,
    spec: PayloadBindingSpec,
    payload_cfg: dict[str, Any],
    config: Any,
    *,
    run_id: str,
) -> ProcessingStage:
    builder = getattr(reader, "build_payload_materialize_stage", None)
    if not callable(builder):
        reader_name = type(reader).__name__ if reader is not None else "<missing reader>"
        msg = (
            "payload_lifecycle requires the source/reader stage to provide "
            "build_payload_materialize_stage(). This keeps payload materialization "
            f"modality-owned instead of hard-coding audio in the central planner; got {reader_name}."
        )
        raise ValueError(msg)
    return builder(
        payload_spec=spec,
        payload_config=payload_cfg,
        pipeline_config=config,
        run_id=run_id,
    )


def _post_release_payload_lifecycle_stage(
    config: Any,
    reader: ProcessingStage | None,
    consumers: list[ProcessingStage],
    primary_spec: PayloadBindingSpec,
    *,
    run_id: str,
) -> ProcessingStage | None:
    if reader is None or not bool(getattr(reader, "enable_global_bucketing", False)):
        return None
    builder = getattr(reader, "build_payload_lifecycle_post_release_stage", None)
    if not callable(builder):
        msg = (
            "Global bucketing is enabled, but the source/reader stage does not provide "
            "build_payload_lifecycle_post_release_stage(). The central payload lifecycle "
            "planner only owns generic insertion order; modality-specific assembly must be "
            f"provided by the planner stage, got {type(reader).__name__}."
        )
        raise ValueError(msg)
    return builder(
        pipeline_config=config,
        consumers=consumers,
        primary_payload_spec=primary_spec,
        run_id=run_id,
    )


def _pipeline_run_id(config: Any) -> str:
    value = _config_get(config, "_curator_pipeline_run_id")
    text = str(value or "").strip()
    if not text:
        msg = "Pipeline config is missing internal _curator_pipeline_run_id"
        raise ValueError(msg)
    return text


def _last_manifest_reader(stages: list[ProcessingStage]) -> ProcessingStage | None:
    readers = [stage for stage in stages if _is_manifest_reader(stage)]
    return readers[-1] if readers else None


def _is_manifest_reader(stage: ProcessingStage) -> bool:
    return callable(getattr(stage, "build_payload_materialize_stage", None))


def _payload_release_stage_class() -> type[ProcessingStage]:
    from nemo_curator.stages.payload_lifecycle import PayloadReleaseStage

    return PayloadReleaseStage


def _config_section(config: Any, key: str) -> dict[str, Any]:
    value = _config_get(config, key, {})
    if value is None:
        return {}
    value = _as_container(value)
    if not isinstance(value, dict):
        msg = f"{key} must be a mapping when configured, got {type(value).__name__}"
        raise TypeError(msg)
    return dict(value)


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    get = getattr(config, "get", None)
    if callable(get):
        return get(key, default)
    return default


def _as_container(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return value


def _normalise_string_list(value: Any, *, key: str) -> list[str]:
    value = _as_container(value)
    if value is None:
        return []
    items = [value] if isinstance(value, str) else list(value)
    result = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            result.append(text)
    if not result:
        msg = f"{key} must contain at least one non-empty value"
        raise ValueError(msg)
    return result


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    if not result:
        msg = "At least one non-empty string is required"
        raise ValueError(msg)
    return result


def _single_selector(value: Any, *, key: str) -> str:
    values = _normalise_string_list(value, key=key)
    if len(values) != 1:
        msg = f"{key} must contain exactly one stage selector, got {values}"
        raise ValueError(msg)
    return values[0]


def _find_stage_index(stages: list[ProcessingStage], selector: str, *, key: str) -> int:
    matches = [idx for idx, stage in enumerate(stages) if selector in _stage_match_idents(stage)]
    if not matches:
        available = sorted({ident for stage in stages for ident in _stage_match_idents(stage)})
        msg = f"{key} selector {selector!r} did not match any stage. Available selectors: {available}"
        raise ValueError(msg)
    if len(matches) > 1:
        names = [stages[idx].name for idx in matches]
        msg = f"{key} selector {selector!r} matched multiple stages: {names}"
        raise ValueError(msg)
    return matches[0]


def _stage_match_idents(stage: ProcessingStage) -> set[str]:
    stage_type = type(stage)
    return {
        ident
        for ident in (
            getattr(stage, "_curator_stage_id", None),
            getattr(stage, "name", None),
            stage_type.__name__,
            f"{stage_type.__module__}.{stage_type.__name__}",
        )
        if ident
    }


def _first_attr(stages: list[ProcessingStage], attr: str, default: Any) -> Any:
    for stage in stages:
        value = getattr(stage, attr, None)
        if value not in (None, ""):
            return value
    return default


def _consumer_payload_key(stages: list[ProcessingStage], attr: str, default: str) -> str:
    return str(_first_attr(stages, attr, default))


def _list_or_derived(payload_cfg: dict[str, Any], key: str, index: int, source_key: str, suffix: str) -> str:
    values = _as_container(payload_cfg.get(key))
    if values:
        values = list(values)
        if index >= len(values):
            msg = f"payload_lifecycle.{key} must contain one value for each payload key"
            raise ValueError(msg)
        return str(values[index])
    return _derived_key(source_key, suffix)


def _list_or_default(payload_cfg: dict[str, Any], key: str, index: int, default: str) -> str:
    values = _as_container(payload_cfg.get(key))
    if values:
        values = list(values)
        if index >= len(values):
            msg = f"payload_lifecycle.{key} must contain one value for each payload key"
            raise ValueError(msg)
        return str(values[index])
    return default


def _derived_key(source_key: str, suffix: str) -> str:
    stem = re.sub(r"(_filepath|_path|_file)$", "", source_key)
    return f"{stem}_{suffix}"


def _materialize_stage_name(
    payload_cfg: dict[str, Any], index: int, source_key: str, *, total: int | None = None
) -> str:
    base = str(payload_cfg.get("materialize_stage_name", "audio_payload_materialize"))
    if total in (None, 1):
        return base
    return f"{base}_{index}_{re.sub(r'[^A-Za-z0-9_]+', '_', source_key)}"


def _validate_unique(values: list[str], label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        else:
            seen.add(value)
    if duplicates:
        msg = f"Duplicate {label}(s): {sorted(duplicates)}"
        raise ValueError(msg)
