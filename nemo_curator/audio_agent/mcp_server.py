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

"""MCP adapter — exposes the audio-agent verbs as typed tools for MCP hosts.

Thin wrapper over the exact same deterministic core as the CLI/`verbs`, so
Claude/Cursor (or any MCP client) get native, typed tools. ``mcp`` is an optional
dependency; the module imports without it and ``main`` prints an install hint.

    pip install "mcp[cli]"
    python -m nemo_curator.audio_agent.mcp_server        # stdio server
"""

from __future__ import annotations

from typing import Any


def build_server() -> Any:  # noqa: ANN401, PLR0915, C901 - a flat tool registry: one statement per verb
    """Construct and return the FastMCP server (imports ``mcp`` lazily)."""
    from mcp.server.fastmcp import FastMCP

    from nemo_curator import audio_agent as aa

    server = FastMCP("nemo-curator-audio-agent")

    # Tool errors are the SDK's job, deliberately not ours. FastMCP wraps anything a tool
    # raises in ``ToolError`` (mcp/server/fastmcp/tools/base.py) and the server turns that
    # into ``CallToolResult(isError=True)``, so a raising verb can never reach the client as
    # a protocol-level exception. A local wrapper that caught the exception and RETURNED an
    # error dict was strictly worse: the call then looks like a success, so ``isError`` stays
    # False and a host keying off that flag cannot see the failure at all.

    @server.tool()
    def discover() -> dict[str, Any]:
        """List agent-ready audio stages with category and one-liner."""
        return aa.discover()

    @server.tool()
    def describe(name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return one stage's contract (and card), resolved against the params you will use.

        Reads and writes follow from params, so pass the ones the recipe will set; without them
        the answer describes the defaults.
        """
        return aa.describe(name, params)

    @server.tool()
    def producers(role: str) -> dict[str, Any]:
        """Return the stages that write a role or key -- "which stage produces segments?"."""
        return aa.producers(role)

    @server.tool()
    def catalog_tree() -> dict[str, Any]:
        """Return the L0 category tree for coarse-to-fine routing."""
        return aa.catalog_tree()

    @server.tool()
    def cards(category: str | None = None, names: list[str] | None = None) -> dict[str, Any]:
        """L1 one-liners for a category, or L2 full cards for named finalists."""
        return aa.cards(category=category, names=names)

    @server.tool()
    def context(
        goal: dict[str, Any] | None = None,
        data: str | None = None,
        stages: list[str] | None = None,
        roles: list[str] | None = None,
        planning_preference: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assemble a PlanningContext (category tree + profile + env + blueprints)."""
        return aa.context(
            goal,
            data=data,
            stages=stages,
            roles=roles,
            planning_preference=planning_preference,
        )

    @server.tool()
    def validate(
        recipe: dict[str, Any],
        data: str | None = None,
        expected_outputs: list[str] | None = None,
        acceptance_criteria: list[dict[str, Any]] | None = None,
        request_type: str | None = None,
    ) -> dict[str, Any]:
        """Validate mechanical runnability and return a grounded Verdict.

        ``acceptance_criteria`` + ``request_type`` add the 1A.1 acceptance checks
        (criterion fields must be producible; request-type sanity). The additive
        ``semantic_review`` packet must be interpreted by the host LLM before smoke;
        a mechanical ``pass`` alone is not intent approval."""
        return aa.validate(
            recipe,
            data=data,
            expected_outputs=expected_outputs,
            acceptance_criteria=acceptance_criteria,
            request_type=request_type,
        )

    @server.tool()
    def smoke(  # noqa: PLR0913
        recipe: dict[str, Any],
        sample: int = 10,
        data: str | None = None,
        output_dir: str | None = None,
        bootstrap_ray: bool = False,
        calibration: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a recipe on a bounded sample and return evidence (incl. a ``smoke_token``).

        ``bootstrap_ray`` auto-starts a local Ray head when none is reachable;
        ``output_dir`` is retained as the verb's legacy no-op; sampled writes are
        always isolated in an ephemeral sandbox. ``calibration`` accepts either
        the mapping from a smoke or the complete wrapper returned by ``calibrate``."""
        return aa.smoke(
            recipe,
            sample=sample,
            data=data,
            output_dir=output_dir,
            bootstrap_ray=bootstrap_ray,
            calibration=calibration,
        )

    @server.tool()
    def run(  # noqa: PLR0913
        recipe: dict[str, Any],
        confirm: bool | str = False,
        data: str | None = None,
        output_dir: str | None = None,
        checkpoint_path: str | None = None,
        bootstrap_ray: bool = False,
        smoke_token: str | None = None,
        calibration: dict[str, Any] | None = None,
        goal: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Confirm-gated full run. Refuses without explicit confirmation.

        ``smoke_token`` satisfies ``AUDIO_AGENT_REQUIRE_SMOKE`` (pass the token from a
        prior ``smoke``); ``bootstrap_ray`` auto-starts Ray; ``checkpoint_path`` enables
        partial-run resume; ``goal`` records what the run was for in provenance.
        ``calibration`` accepts the complete wrapper returned by ``calibrate``; omit it
        and the measurements a prior ``smoke`` of this exact recipe stored are applied
        automatically (the resource plan says so in its notes).
        ``output_dir`` is retained as the verb's legacy no-op; configure output
        paths on recipe stages."""
        return aa.run(
            recipe,
            confirm=confirm,
            data=data,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            bootstrap_ray=bootstrap_ray,
            smoke_token=smoke_token,
            calibration=calibration,
            goal=goal,
        )

    @server.tool()
    def report(
        output: str,
        recipe: dict[str, Any] | None = None,
        data: str | None = None,
    ) -> dict[str, Any]:
        """Post-hoc evidence report from an output manifest/dir.

        Supplying ``recipe`` binds the evidence to its terminal serializer,
        frozen identity, and acceptance contract. In that form ``data`` is only
        a consistency assertion about the recipe's configured source."""
        return aa.report(output, recipe=recipe, data=data)

    @server.tool()
    def verify(
        acceptance_criteria: list[dict[str, Any]],
        evidence: dict[str, Any] | None = None,
        frozen_criteria: list[dict[str, Any]] | None = None,
        recipe: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Verify acceptance criteria against evidence -> AcceptanceReport (1A.1/1A.3).

        ``overall`` is ``met`` iff every ``must`` criterion is met; states are
        met / not_met / unverifiable / unachievable (never silently relaxed). Pass
        ``frozen_criteria`` or a ``recipe`` to run the honesty guard (flags a
        weaker-than-confirmed contract and forces overall=not_met)."""
        return aa.verify(acceptance_criteria, evidence=evidence, frozen_criteria=frozen_criteria, recipe=recipe)

    @server.tool()
    def resolve(
        stage: str,
        label: str | None = None,
        use_case: str | None = None,
        explicit: dict[str, Any] | None = None,
        data: str | None = None,
    ) -> dict[str, Any]:
        """Resolve an outcome (label/use_case/explicit) to concrete stage config (1A.2).

        Maps a user-facing outcome to params (or a PreserveByValueStage filter) via
        the card's metrics anchors/presets, with an auditable strategy trail. Never
        exposes or invents internal thresholds."""
        return aa.resolve(stage, label=label, use_case=use_case, explicit=explicit, data=data)

    @server.tool()
    def runs(  # noqa: PLR0913
        run_id: str | None = None,
        data: str | None = None,
        stage: str | None = None,
        since: str | None = None,
        limit: int = 50,
        goal: dict[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        """List local run records (provenance), or load one by run_id. Local history,
        not shared memory/learning. Filter by dataset path/key, stage, and time.

        With a folder ``data`` path and ``goal`` (the user's current request), ranks
        priors by how much of that request is covered by each prior's recorded prompt
        plus ``pipeline_summary`` — call this before inventing a recipe.
        """
        return aa.runs(run_id=run_id, data=data, stage=stage, since=since, limit=limit, goal=goal)

    @server.tool()
    def reuse_scan(
        recipe: dict[str, Any],
        data: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Find prior artifacts this recipe could reuse without changing state."""
        return aa.reuse_scan(recipe, data=data, limit=limit)

    @server.tool()
    def delta_run(  # noqa: PLR0913
        recipe: dict[str, Any] | None = None,
        from_run: str | None = None,
        data: str | None = None,
        confirm: bool | str = False,
        bootstrap_ray: bool = False,
        smoke_token: str | None = None,
        calibration: dict[str, Any] | None = None,
        goal: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run only the files that changed since a prior run of this recipe, and merge them in.

        Without ``confirm`` this returns the card (which files moved, which manifests would be
        rewritten, how many rows survive, what it saves). Everything a delta relies on is
        checked first and refused by name, so a ``no_delta`` answer means run normally rather
        than that a partial result was accepted.

        ``from_run`` adopts that run's own recipe (a run_id from ``runs`` or from a scan's
        ``prior_on_same_path``) instead of passing one, so the delta matches the pipeline whose
        artifacts it resumes from rather than a retyped near-copy of it."""
        return aa.delta_run(
            recipe,
            from_run=from_run,
            data=data,
            confirm=confirm,
            bootstrap_ray=bootstrap_ray,
            smoke_token=smoke_token,
            calibration=calibration,
            goal=goal,
        )

    @server.tool()
    def add_checkpoint(
        recipe: dict[str, Any],
        data: str | None = None,
        output_path: str | None = None,
        after: str | None = None,
    ) -> dict[str, Any]:
        """Say where a mid-pipeline manifest would make the expensive stages reusable, and
        return the recipe carrying it (nothing is written or run).

        With ``data`` the checkpoint location is derived; never ask the user for one.
        ``output_path`` overrides it, and is for a user who asked to keep the metadata
        somewhere of their own.
        """
        return aa.add_checkpoint(recipe, data=data, output_path=output_path, after=after)

    @server.tool()
    def plan_checkpoint(  # noqa: PLR0913 - mirrors the public checkpoint policy surface
        recipe: dict[str, Any] | None = None,
        from_run: str | None = None,
        data: str | None = None,
        output_path: str | None = None,
        decision_stage: str | None = None,
        decision_value: Any = None,  # noqa: ANN401 - card-declared scalar/categorical value
        decision_conditions: Any = None,  # noqa: ANN401 - complete card-declared compound surface
        choice: str | None = None,
        retention_sec: int = 0,
        owner: str = "user",
    ) -> dict[str, Any]:
        """Build a complete same-dataset checkpoint candidate before authoritative smoke.

        Use ``recipe`` for the first run. Use ``from_run`` plus ``decision_stage`` and
        ``decision_value`` for scalar feedback, or ``decision_conditions`` for
        a complete card-declared compound ge condition set. When ``data``
        changed, the result routes to existing delta/fresh behavior rather than
        combining both kinds of reuse.

        With ``data`` the checkpoint location is derived; never ask the user for one.
        ``output_path`` overrides it, and is for a user who asked to keep the metadata
        somewhere of their own.
        """
        return aa.plan_checkpoint(
            recipe,
            from_run=from_run,
            data=data,
            output_path=output_path,
            decision_stage=decision_stage,
            decision_value=decision_value,
            decision_conditions=decision_conditions,
            choice=choice,
            retention_sec=retention_sec,
            owner=owner,
        )

    @server.tool()
    def checkpoints(gc: bool = False) -> dict[str, Any]:
        """List the managed checkpoint cache; with ``gc`` collect what nothing can reuse.

        ``gc`` deletes only orphaned and expired entries inside the managed directory. A
        reusable checkpoint, and any path the user chose themselves, are never candidates.
        """
        return aa.checkpoints(gc=gc)

    @server.tool()
    def reindex() -> dict[str, Any]:
        """Rebuild the run/artifact lookup index from its JSON source records."""
        return aa.reindex()

    @server.tool()
    def plan_continuation(  # noqa: PLR0913
        recipe: dict[str, Any],
        parent_run_id: str | None = None,
        data: str | None = None,
        execute: bool = False,
        choice: str | None = None,
        confirm: bool | str = False,
        output_dir: str | None = None,
        checkpoint_path: str | None = None,
        bootstrap_ray: bool = False,
        smoke_token: str | None = None,
        calibration: dict[str, Any] | None = None,
        goal: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Plan, or explicitly execute, a safe reuse choice for a follow-up recipe.

        ``parent_run_id`` is optional because the artifact scan can find reusable
        work independently. Execution remains subject to the normal confirmation
        and smoke-evidence gates. ``calibration`` accepts the complete wrapper
        returned by ``calibrate``."""
        return aa.plan_continuation(
            recipe,
            parent_run_id,
            data=data,
            execute=execute,
            choice=choice,
            confirm=confirm,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            bootstrap_ray=bootstrap_ray,
            smoke_token=smoke_token,
            calibration=calibration,
            goal=goal,
        )

    @server.tool()
    def calibrate(smoke_report: dict[str, Any]) -> dict[str, Any]:
        """Extract measured per-stage resources from a smoke report (1C.2).

        The result is a ``{"calibration": {...}}`` wrapper. Pass that complete
        result unchanged as ``calibration`` to ``smoke``, ``run``, or
        ``plan_continuation``; the core accepts both this wrapper and a bare
        stage-to-measurements mapping."""
        return aa.calibrate(smoke_report)

    @server.tool()
    def diagnose(  # noqa: PLR0913
        error: str,
        recipe: dict[str, Any] | None = None,
        operation: str = "run",
        phase: str = "runtime",
        attempted_actions: list[str] | None = None,
        execution_target: str | None = None,
    ) -> dict[str, Any]:
        """Analyze a failure and return evidence, grounded choices, and a user-decision prompt.

        This tool never applies a fix. The host should explain the relevant facts,
        recommend an available option against the user's constraints, and ask before
        any environment, host, credential, launch, device, or recipe change."""
        return aa.diagnose(
            error,
            recipe=recipe,
            operation=operation,
            phase=phase,
            attempted_actions=attempted_actions,
            execution_target=execution_target,
        )

    @server.tool()
    def doctor() -> dict[str, Any]:
        """Return machine health plus structured, non-executing remediation options."""
        return aa.doctor()

    @server.tool()
    def install_skill(  # noqa: PLR0913
        scope: str = "project",
        host: str = "all",
        mode: str = "copy",
        skills: list[str] | None = None,
        dest: str | None = None,
        force: bool = False,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Install the packaged audio skills into the host discovery directories.

        ``dry_run`` defaults to true here, unlike the CLI: this tool writes files
        outside the curation flow, so an MCP host must ask for the write explicitly
        rather than get it from a tool call that reads like an inspection. Refuses to
        replace a target whose content differs unless ``force`` is set."""
        return aa.install_skill(
            scope=scope,
            host=host,
            mode=mode,
            skills=skills,
            dest=dest,
            force=force,
            dry_run=dry_run,
        )

    return server


def main() -> int:
    try:
        server = build_server()
    except ModuleNotFoundError:
        print("MCP is not installed. Install it with:  pip install 'mcp[cli]'")
        return 1
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
