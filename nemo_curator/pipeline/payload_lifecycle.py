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
# ruff: noqa: ANN401, ARG001, BLE001, PLR2004, S110, TRY004

"""Generic pipeline expansion for payload handle lifecycles."""

from __future__ import annotations

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
        required = [
            stage.name for stage in stages if bool(getattr(stage, "_curator_requires_payload_lifecycle", False))
        ]
        if required:
            msg = f"Stage(s) {required} require payload_lifecycle.enabled=true"
            raise ValueError(msg)
        return stages

    payload_release_stage_cls = _payload_release_stage_class()
    materialize_idx, release_idx, consumers = _payload_lifecycle_positions(stages, payload_cfg)
    run_id = _pipeline_run_id(config)

    reader = _last_manifest_reader(stages[: materialize_idx + 1])
    payload_spec = _payload_binding_spec(payload_cfg, stages=stages, consumers=consumers, reader=reader)
    _configure_planned_source_segment_inputs(reader, payload_cfg, payload_spec, config)
    _validate_payload_consumers(consumers, payload_spec)
    planner_owner = _validate_single_segment_planner_owner(
        reader,
        consumers,
        config=config,
    )

    materializer = _build_payload_materializer(reader, payload_spec, payload_cfg, config, run_id=run_id)
    release = payload_release_stage_cls(
        name=str(payload_cfg.get("release_stage_name", "payload_release")),
        payload_ref_key=payload_spec.ref_key,
        waveform_key=payload_spec.waveform_key,
    )

    assembler = _post_release_payload_lifecycle_stage(config, reader, consumers, payload_spec, run_id=run_id)
    execution_source = _payload_lifecycle_source_stage(reader)
    dispatch_unpack = _dispatch_batch_unpack_stage(
        execution_source,
        planner_owner,
        stages=stages,
        materialize_idx=materialize_idx,
    )
    _configure_lifecycle_helpers(
        stages,
        materializer=materializer,
        dispatch_unpack=dispatch_unpack,
        release=release,
        assembler=assembler,
    )
    _configure_payload_tracking(
        stages[materialize_idx + 1 : release_idx + 1],
        materializer=materializer,
        ref_key=payload_spec.ref_key,
        preserve_terminals=assembler is not None,
    )
    expanded = _expanded_lifecycle_graph(
        stages,
        reader=reader,
        execution_source=execution_source,
        materialize_idx=materialize_idx,
        materializer=materializer,
        planner_owner=planner_owner,
        dispatch_unpack=dispatch_unpack,
        release_idx=release_idx,
        release=release,
        assembler=assembler,
    )
    logger.info(
        "Expanded logical graph into payload lifecycle execution graph: {}",
        " -> ".join(stage.name for stage in expanded),
    )
    return expanded


def _configure_lifecycle_helpers(
    stages: list[ProcessingStage],
    *,
    materializer: ProcessingStage,
    dispatch_unpack: ProcessingStage | None,
    release: ProcessingStage,
    assembler: ProcessingStage | None,
) -> None:
    extended_metrics = any(bool(getattr(stage, "extended_performance_metrics", False)) for stage in stages)
    helpers = [materializer, release]
    helpers.extend(stage for stage in (dispatch_unpack, assembler) if stage is not None)
    for helper in helpers:
        helper.extended_performance_metrics = extended_metrics


def _configure_payload_tracking(
    consumers: list[ProcessingStage],
    *,
    materializer: ProcessingStage,
    ref_key: str,
    preserve_terminals: bool,
) -> None:
    """Enable lifecycle bookkeeping only on stages that can receive refs."""
    for stage in consumers:
        stage._curator_tracks_payload_refs = True
        stage._curator_payload_ref_key = ref_key
    if preserve_terminals:
        for stage in [materializer, *consumers]:
            stage._curator_preserves_terminal_tasks = True


def _expanded_lifecycle_graph(  # noqa: PLR0913
    stages: list[ProcessingStage],
    *,
    reader: ProcessingStage | None,
    execution_source: ProcessingStage | None,
    materialize_idx: int,
    materializer: ProcessingStage,
    planner_owner: ProcessingStage | None,
    dispatch_unpack: ProcessingStage | None,
    release_idx: int,
    release: ProcessingStage,
    assembler: ProcessingStage | None,
) -> list[ProcessingStage]:
    expanded: list[ProcessingStage] = []
    for idx, stage in enumerate(stages):
        expanded.append(execution_source if stage is reader else stage)
        if idx == materialize_idx:
            expanded.append(materializer)
        if dispatch_unpack is not None and stage is planner_owner:
            expanded.append(dispatch_unpack)
        if idx == release_idx:
            expanded.extend([release, *([assembler] if assembler is not None else [])])
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


def _dispatch_batch_unpack_stage(
    execution_source: ProcessingStage | None,
    owner: ProcessingStage | None,
    *,
    stages: list[ProcessingStage],
    materialize_idx: int,
) -> ProcessingStage | None:
    """Build an unpack helper only for sources that emit atomic dispatch rows."""
    if execution_source is None or not bool(getattr(execution_source, "_curator_emits_dispatch_batches", False)):
        return None
    if owner is None:
        msg = "A dispatch-batch source requires one selected owner stage"
        raise ValueError(msg)
    if not bool(getattr(owner, "_curator_accepts_dispatch_batches", False)):
        msg = f"Dispatch-batch owner {owner.name!r} does not implement exact dispatch-batch consumption"
        raise TypeError(msg)
    validate_source = getattr(owner, "validate_dispatch_source", None)
    if not callable(validate_source):
        msg = f"Dispatch-batch owner {owner.name!r} does not expose source-contract validation"
        raise TypeError(msg)
    validate_source(execution_source)
    owner_idx = stages.index(owner)
    if owner_idx != materialize_idx + 1:
        between = [stage.name for stage in stages[materialize_idx + 1 : owner_idx]]
        msg = (
            "The global dispatch owner must be the first logical stage after payload materialization so its "
            f"atomic batches cannot be altered; found intermediate stage(s): {between}"
        )
        raise ValueError(msg)
    dispatch_window_size = int(getattr(execution_source, "_curator_dispatch_window_size", 2))
    owner.batch_size = dispatch_window_size
    logger.info(
        "Configured dispatch-batch owner {} with backend window size {}",
        owner.name,
        dispatch_window_size,
    )

    from nemo_curator.stages.dispatch_batch import DispatchBatchUnpackStage

    return DispatchBatchUnpackStage(name=f"dispatch_batch_unpack_after_{owner.name}")


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


def _payload_binding_spec(
    payload_cfg: dict[str, Any],
    *,
    stages: list[ProcessingStage],
    consumers: list[ProcessingStage],
    reader: ProcessingStage | None,
) -> PayloadBindingSpec:
    unsupported = [
        key
        for key in (
            "payloads",
            "payload_keys",
            "ref_keys",
            "waveform_keys",
            "sample_rate_keys",
            "num_samples_keys",
            "duration_keys",
        )
        if key in payload_cfg
    ]
    if unsupported:
        msg = (
            "payload_lifecycle supports exactly one payload source. Replace plural configuration "
            f"{unsupported} with singular source_key/ref_key/waveform_key fields."
        )
        raise ValueError(msg)
    removed_lease_keys = [key for key in ("lease_ttl_s", "materialized_lease_ttl_s") if key in payload_cfg]
    if removed_lease_keys:
        msg = (
            f"Removed payload lifecycle lease config {removed_lease_keys}. "
            "Reservation expiry and explicit published-payload release are internal lifecycle policy."
        )
        raise ValueError(msg)
    source_key = str(payload_cfg.get("source_key", "audio_filepath")).strip()
    if not source_key:
        msg = "payload_lifecycle.source_key must be non-empty"
        raise ValueError(msg)
    return PayloadBindingSpec(
        source_key=source_key,
        ref_key=str(
            payload_cfg.get("ref_key") or _consumer_payload_key(consumers, "waveform_ref_key", "waveform_ref")
        ),
        waveform_key=str(
            payload_cfg.get("waveform_key") or _consumer_payload_key(consumers, "waveform_key", "waveform")
        ),
        sample_rate_key=str(
            payload_cfg.get("sample_rate_key") or _consumer_payload_key(consumers, "sample_rate_key", "sample_rate")
        ),
        num_samples_key=str(
            payload_cfg.get("num_samples_key") or _first_attr(stages, "num_samples_key", "num_samples")
        ),
        duration_key=str(
            payload_cfg.get("duration_key")
            or _first_attr(stages, "duration_key", getattr(reader, "duration_key", "duration"))
        ),
        materialize_stage_name=str(payload_cfg.get("materialize_stage_name", "audio_payload_materialize")),
    )


def _validate_payload_consumers(consumers: list[ProcessingStage], payload_spec: PayloadBindingSpec) -> None:
    for stage in consumers:
        if not hasattr(stage, "resolve_payload_refs_for_batch"):
            msg = (
                f"Payload consumer {stage.name!r} is not payload-aware. Stages that consume payload refs must "
                "implement PayloadAwareStageMixin or an equivalent resolve_payload_refs_for_batch() contract."
            )
            raise TypeError(msg)
        binding = _stage_payload_binding(stage)
        if not binding:
            msg = f"Payload consumer {stage.name!r} does not declare a payload ref binding"
            raise ValueError(msg)
        ref_key = str(binding["ref_key"])
        if ref_key != payload_spec.ref_key:
            msg = (
                f"Payload consumer {stage.name!r} declares ref_key={ref_key!r}, but payload_lifecycle "
                f"materializes {payload_spec.ref_key!r}"
            )
            raise ValueError(msg)
        waveform_key = str(binding.get("waveform_key") or "")
        if waveform_key != payload_spec.waveform_key:
            msg = (
                f"Payload consumer {stage.name!r} must declare waveform_key={payload_spec.waveform_key!r} "
                f"for ref_key={ref_key!r}; got {waveform_key!r}"
            )
            raise ValueError(msg)


def _stage_payload_binding(stage: ProcessingStage) -> dict[str, str] | None:
    binding = getattr(stage, "payload_binding", None)
    if callable(binding):
        binding = binding()
    if binding is not None:
        if not isinstance(binding, dict):
            msg = f"{stage.name}.payload_binding must return a mapping"
            raise TypeError(msg)
        ref_key = str(binding.get("ref_key") or "").strip()
        waveform_key = str(binding.get("waveform_key") or "").strip()
        if not ref_key or not waveform_key:
            msg = f"{stage.name}.payload_binding requires non-empty ref_key and waveform_key"
            raise ValueError(msg)
        return {**{str(k): str(v) for k, v in binding.items()}, "ref_key": ref_key, "waveform_key": waveform_key}
    ref_key = getattr(stage, "waveform_ref_key", None)
    waveform_key = getattr(stage, "waveform_key", None)
    if ref_key and waveform_key:
        return {
            "ref_key": str(ref_key),
            "waveform_key": str(waveform_key),
            "sample_rate_key": str(getattr(stage, "sample_rate_key", "sample_rate")),
            "num_samples_key": str(getattr(stage, "num_samples_key", "num_samples")),
        }
    return None


def _configure_planned_source_segment_inputs(
    reader: ProcessingStage | None,
    payload_cfg: dict[str, Any],
    payload_spec: PayloadBindingSpec,
    config: Any,
) -> None:
    if reader is None or not bool(getattr(reader, "enable_global_bucketing", False)):
        return
    scheduler_cfg = _config_section(config, "global_audio_scheduler")
    configured = scheduler_cfg.get("segment_input_keys", payload_cfg.get("segment_input_keys"))
    segment_input_keys: list[str] = []
    if configured is not None:
        segment_input_keys.extend(_normalise_string_list(configured, key="global_audio_scheduler.segment_input_keys"))
    segment_input_keys.append(payload_spec.source_key)
    reader.segment_input_keys = _dedupe_strings(segment_input_keys)
    reader.run_id = _pipeline_run_id(config)
    if "parent_store_actor_name_prefix" in scheduler_cfg:
        reader.parent_store_actor_name_prefix = str(scheduler_cfg["parent_store_actor_name_prefix"])


def _validate_single_segment_planner_owner(
    reader: ProcessingStage | None,
    consumers: list[ProcessingStage],
    *,
    config: Any,
) -> ProcessingStage | None:
    if reader is None or not bool(getattr(reader, "enable_global_bucketing", False)):
        return None
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
    return matching_consumers[0]


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
    payload_spec: PayloadBindingSpec,
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
        payload_spec=payload_spec,
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
