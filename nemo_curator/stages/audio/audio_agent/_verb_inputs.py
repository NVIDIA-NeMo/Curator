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

"""Recipe binding, source profiling, and input refusals."""

from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlsplit

from nemo_curator.stages.audio.audio_agent import profiler
from nemo_curator.stages.audio.audio_agent.contracts import Issue
from nemo_curator.stages.audio.audio_agent.recipe import (
    Recipe,
)


def _ray_address_is_local(address: str | None) -> bool:  # noqa: PLR0911
    """Whether a configured Ray address names this machine.

    An unset address and Ray's ``local`` mode retain local checks. A head
    THIS process bootstrapped is always local: it runs on this machine even when Ray
    advertises the node's LAN IP (multi-NIC / cloud), so its probed VRAM/GPU names are
    valid. Otherwise a concrete address is local only when its host is ``localhost`` or an
    IPv4/IPv6 loopback address; unknown hostnames fail closed as remote so driver
    environment facts are never projected onto another machine.
    """
    value = str(address or "").strip()
    if not value or value.casefold() == "local":
        return True
    try:  # a self-started head is on this machine regardless of the advertised IP
        from nemo_curator.stages.audio.audio_agent import _ray

        if _ray.owns_cluster(value):
            return True
    except Exception:  # noqa: BLE001, S110 - ownership is best-effort; fall through to host checks
        pass
    if value.casefold().rstrip(".") in {"localhost", "::1", "[::1]"}:
        return True
    try:
        parsed = urlsplit(value if "://" in value else f"//{value}")
        host = (parsed.hostname or "").casefold().rstrip(".")
    except ValueError:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _ray_execution_address(address: str | None) -> str | None:
    """Resolve a pre-existing connection without starting or replacing a cluster."""
    if address and address.strip().casefold() != "auto":
        return address
    try:
        import ray
    except ImportError:
        return address
    if ray.is_initialized():
        from nemo_curator.stages.audio.audio_agent._ray import _connected_address

        # An initialized but unidentified connection must not inherit driver facts.
        return _connected_address(ray) or "auto"
    return address


def _ray_execution_target(address: str | None) -> str:
    """Map a Ray address to the environment-analysis target vocabulary."""
    return "local" if _ray_address_is_local(_ray_execution_address(address)) else "external_ray"


def _as_recipe(recipe: Recipe | dict[str, Any]) -> Recipe:
    rec = recipe if isinstance(recipe, Recipe) else Recipe.from_dict(recipe)
    # ``Recipe(...)`` remains a convenient SDK constructor and can bypass
    # Recipe.from_dict. Hold that path to the same fail-closed contract boundary
    # before validate/smoke/run/reuse performs any I/O.
    from nemo_curator.stages.audio.audio_agent.acceptance import parse_criteria
    from nemo_curator.stages.audio.audio_agent.recipe import parse_planning_preference

    parse_criteria(rec.acceptance_criteria)
    rec.planning_preference = parse_planning_preference(rec.planning_preference)
    return rec


def _dataset_binding(rec: Recipe, data: str | None):  # noqa: ANN202 - private adapter type
    """Resolve the recipe's physical input; stage params remain execution truth."""
    from nemo_curator.stages.audio.audio_agent.input_identity import resolve_dataset_binding

    return resolve_dataset_binding(rec, data)


def _binding_blocks_execution(binding: Any, data: str | None) -> bool:  # noqa: ANN401
    """Whether a binding problem makes execution unsafe rather than merely unkeyed."""
    if binding.status in {"missing", "mismatch", "unsupported"}:
        return True
    if binding.status != "ambiguous":
        return False
    # A configured multi-manifest reader can still execute exactly as authored
    # when no singular --data assertion pretends to describe the whole set. It
    # simply gets no reuse identity until aggregate profiling exists.
    return not (
        data is None
        and binding.source_ref == "ManifestReader"
        and binding.source_index == 0
        and len(binding.configured_paths) > 1
    )


def _profile_binding(binding: Any):  # noqa: ANN202, ANN401 - DataProfile | None without eager imports
    """Profile only the source the recipe actually executes, never caller text."""
    if binding.status not in {"resolved", "ambiguous"} or not binding.profile_source:
        return None
    # Source adapters own these options. Silently dropping one changes which inputs
    # the identity covers while execution still honors the authored selection.
    prof = profiler.profile_data(binding.profile_source, **binding.profile_kwargs)
    # A lexical remote selector or unsupported selector must not become a
    # reusable shape key merely because DataProfile can hash its spelling.
    return prof if prof.kind != "unknown" or prof.source_errors else None


def _profile_error(profile: Any) -> str:  # noqa: ANN401
    """Fatal source-definition error found by the read-only profiler."""
    errors = list(getattr(profile, "source_errors", None) or [])
    return "; ".join(str(error) for error in errors)


def _profile_refusal(binding: Any, profile: Any) -> dict[str, Any]:  # noqa: ANN401
    return {
        "status": "refused",
        "reason": f"recipe data source is unreadable or malformed: {_profile_error(profile)}",
        "data_binding": binding.to_dict(),
        "data_profile": profile.to_dict(),
    }


def _binding_issue(binding: Any, data: str | None) -> Issue | None:  # noqa: ANN401
    """Translate source binding state into the validation issue vocabulary."""
    if binding.status == "resolved":
        return None
    blocking = _binding_blocks_execution(binding, data)
    code = {
        "missing": "data_source_missing",
        "mismatch": "data_source_mismatch",
        "ambiguous": "data_source_ambiguous",
        "unsupported": "data_source_unsupported",
    }.get(binding.status, "data_source_unresolved")
    return Issue(
        code,
        "error" if blocking else "warning",
        binding.reason,
        stage_index=binding.source_index,
        stage=binding.source_ref,
        fix={
            "missing": "configure an existing source in the first supported source stage",
            "mismatch": ("make --data and Recipe.inputs agree with the first stage's configured source"),
            "unsupported": ("use a supported first source stage/path form or add an explicit source adapter"),
            "ambiguous": (
                "remove the extra source, or omit singular --data for an authored multi-manifest reader"
                if blocking
                else "omit singular --data or use one manifest selector if reusable identity is required"
            ),
        }.get(binding.status, "bind one supported source as the recipe's first stage"),
    )


def _binding_refusal(binding: Any) -> dict[str, Any]:  # noqa: ANN401
    return {
        "status": "refused",
        "reason": f"recipe data source is not safely bound: {binding.reason}",
        "data_binding": binding.to_dict(),
    }
