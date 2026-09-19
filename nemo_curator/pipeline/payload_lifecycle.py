# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic pipeline graph expansion for payload lifecycles."""

from __future__ import annotations

import importlib
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from nemo_curator.stages.base import ProcessingStage

_DEFAULT_MATERIALIZER_TARGET = "nemo_curator.stages.payload_lifecycle.AudioPayloadMaterializeStage"


@dataclass(frozen=True)
class PayloadBindingSpec:
    ref_key: str
    payload_key: str
    sample_rate_key: str
    num_samples_key: str


def payload_lifecycle_enabled(config: dict[str, Any]) -> bool:
    """Report whether a pipeline config turns the payload lifecycle on."""
    lifecycle = config.get("payload_lifecycle") or {}
    if not isinstance(lifecycle, dict):
        msg = "payload_lifecycle must be a mapping"
        raise TypeError(msg)
    return bool(lifecycle.get("enabled", False))


def cleanup_stage_run_resources(stages: list[ProcessingStage]) -> None:
    """Release run-scoped resources created by payload helper stages.

    Detached Ray actors must be removed while the executor still has a live
    Ray connection, so the Ray Data executor invokes this payload-owned helper
    immediately before ``ray.shutdown()``.
    """
    for stage in reversed(stages):
        cleanup = getattr(stage, "cleanup_run_resources", None)
        if not callable(cleanup):
            continue
        try:
            cleanup()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Run-scoped cleanup failed for stage {}: {}", stage, exc)


def expand_payload_lifecycle_stages(
    stages: list[ProcessingStage],
    config: dict[str, Any],
) -> list[ProcessingStage]:
    """Insert modality-owned materialization and generic release stages."""
    if not payload_lifecycle_enabled(config):
        return stages
    lifecycle = config["payload_lifecycle"]

    materialize_after = _required_selector(lifecycle, "materialize_after")
    release_after = _required_selector(lifecycle, "release_after")
    consumer_selectors = _string_list(lifecycle.get("consumers"), key="consumers")
    materialize_index = _stage_index(stages, materialize_after, key="materialize_after")
    release_index = _stage_index(stages, release_after, key="release_after")
    if release_index <= materialize_index:
        msg = "payload_lifecycle.release_after must follow materialize_after"
        raise ValueError(msg)

    consumers = [_stage_for_selector(stages, selector, key="consumers") for selector in consumer_selectors]
    for consumer in consumers:
        index = stages.index(consumer)
        if not materialize_index < index <= release_index:
            msg = f"Payload consumer {consumer.name!r} must be inside the materialize/release window"
            raise ValueError(msg)

    spec = PayloadBindingSpec(
        ref_key=str(lifecycle.get("ref_key", "waveform_ref")),
        payload_key=str(lifecycle.get("payload_key", "waveform")),
        sample_rate_key=str(lifecycle.get("sample_rate_key", "sample_rate")),
        num_samples_key=str(lifecycle.get("num_samples_key", "num_samples")),
    )
    owner = stages[materialize_index]
    run_id = str(config.setdefault("_curator_payload_run_id", uuid.uuid4().hex))
    materializer = _build_materializer(owner, spec, lifecycle, run_id=run_id)

    from nemo_curator.stages.payload_lifecycle import PayloadReleaseStage

    release = PayloadReleaseStage(
        name=str(lifecycle.get("release_stage_name", "payload_release")),
        payload_ref_key=spec.ref_key,
        payload_key=spec.payload_key,
    )
    planned_consumers = _prepare_consumers(consumers, spec)
    replacements = {id(original): planned for original, planned in zip(consumers, planned_consumers, strict=True)}

    expanded: list[ProcessingStage] = []
    for index, stage in enumerate(stages):
        expanded.append(replacements.get(id(stage), stage))
        if index == materialize_index:
            expanded.append(materializer)
        if index == release_index:
            expanded.append(release)
    logger.info("Expanded payload lifecycle graph: {}", " -> ".join(stage.name for stage in expanded))
    return expanded


def _build_materializer(
    owner: ProcessingStage,
    spec: PayloadBindingSpec,
    lifecycle: dict[str, Any],
    *,
    run_id: str,
) -> ProcessingStage:
    target = lifecycle.get("materializer_target")
    if target is not None:
        builder = _load_materializer_target(str(target))
    else:
        owner_builder = getattr(owner, "build_payload_materialize_stage", None)
        builder = owner_builder if callable(owner_builder) else _load_materializer_target(_DEFAULT_MATERIALIZER_TARGET)
    config_factory = getattr(builder, "from_payload_config", builder)
    materializer = config_factory(payload_spec=spec, payload_config=lifecycle, run_id=run_id)
    from nemo_curator.stages.base import ProcessingStage

    if not isinstance(materializer, ProcessingStage):
        msg = f"Payload materializer target returned {type(materializer).__name__}, expected ProcessingStage"
        raise TypeError(msg)
    return materializer


def _load_materializer_target(target: str) -> Callable[..., ProcessingStage]:
    module_name, separator, attribute = target.partition(":")
    if not separator:
        module_name, separator, attribute = target.rpartition(".")
    if not module_name or not attribute:
        msg = f"payload_lifecycle.materializer_target must be a dotted callable, got {target!r}"
        raise ValueError(msg)
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        msg = f"payload_lifecycle.materializer_target {target!r} is not callable"
        raise TypeError(msg)
    return cast("Callable[..., ProcessingStage]", factory)


def _prepare_consumers(
    consumers: list[ProcessingStage],
    spec: PayloadBindingSpec,
) -> list[ProcessingStage]:
    expected = {
        "ref_key": spec.ref_key,
        "payload_key": spec.payload_key,
        "sample_rate_key": spec.sample_rate_key,
        "num_samples_key": spec.num_samples_key,
    }
    from nemo_curator.stages.payload_lifecycle import PayloadResolvingStage

    planned: list[ProcessingStage] = []
    for consumer in consumers:
        binding_provider = getattr(consumer, "payload_binding", None)
        binding = binding_provider() if callable(binding_provider) else None
        if binding is None:
            planned.append(
                PayloadResolvingStage(
                    wrapped_stage=consumer,
                    waveform_ref_key=spec.ref_key,
                    waveform_key=spec.payload_key,
                    sample_rate_key=spec.sample_rate_key,
                    num_samples_key=spec.num_samples_key,
                )
            )
            continue
        if binding != expected:
            msg = f"Payload consumer {consumer.name!r} binding {binding} does not match lifecycle {expected}"
            raise ValueError(msg)
        planned.append(consumer)
    return planned


def _required_selector(config: dict[str, Any], key: str) -> str:
    values = _string_list(config.get(key), key=key)
    if len(values) != 1:
        msg = f"payload_lifecycle.{key} must contain exactly one selector"
        raise ValueError(msg)
    return values[0]


def _string_list(value: object, *, key: str) -> list[str]:
    values = [value] if isinstance(value, str) else list(value or [])
    result = [str(item).strip() for item in values if str(item).strip()]
    if not result:
        msg = f"payload_lifecycle.{key} must contain at least one selector"
        raise ValueError(msg)
    return result


def _stage_index(stages: list[ProcessingStage], selector: str, *, key: str) -> int:
    return stages.index(_stage_for_selector(stages, selector, key=key))


def _stage_for_selector(stages: list[ProcessingStage], selector: str, *, key: str) -> ProcessingStage:
    matches = [stage for stage in stages if selector in _stage_identifiers(stage)]
    if not matches:
        msg = f"payload_lifecycle.{key} selector {selector!r} did not match a stage"
        raise ValueError(msg)
    if len(matches) > 1:
        msg = f"payload_lifecycle.{key} selector {selector!r} matched multiple stages"
        raise ValueError(msg)
    return matches[0]


def _stage_identifiers(stage: ProcessingStage) -> set[str]:
    stage_type = type(stage)
    return {
        str(stage.name),
        stage_type.__name__,
        f"{stage_type.__module__}.{stage_type.__name__}",
    }
