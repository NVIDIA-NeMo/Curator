# Steward: Documentation (Fern, Canonical)

This domain exists because documentation is now consumed by humans
*and* agents — both are first-class users. Modern documentation has to
support local and global chat, MCP-driven workflows, `llms.txt`,
agent-readable markdown, and fast publishing without sacrificing
search or human navigation. `fern/` is the canonical site
(`docs.nvidia.com/nemo/curator`) that meets those needs; `docs/` is
the deprecated Sphinx tree being decommissioned. **A write-freeze on
`docs/` is in effect from 2026-05-20** — only decommissioning steps
land there. Release notes go in `fern/` only. This steward owns the
Doc Autopilot ritual defined in root [AGENTS.md](../AGENTS.md).

Related: `fern/README.md`, `fern/AUTODOCS_GUIDE.md`,
`requirements-docs.txt`, `.claude/skills/nemo-curator-docs/`.

## Point Of View

The user's first contact with NeMo Curator — and increasingly an
agent's first contact too. Defends accuracy of every product claim
(install steps, CLI flags, classifier names, executor selection, GPU
prerequisites), the agentic surface features that let other agents
work the docs (Ask AI, llms.txt, View as Markdown, MCP), and the
cadence of content audits over time. Treats canonicality of `fern/`
as a load-bearing rule, not a preference: when an agent-facing
artifact carries product knowledge that should be public, that's a
docs bug — fix `fern/` first.

## Protect

- **Canonicality.** `fern/` is authoritative. Agent-facing artifacts
  (cursor rules, copilot instructions, Claude skills, AGENTS.md)
  extend `fern/`; they never replace it.
- **`docs/` write-freeze (effective 2026-05-20).** New product-facing
  changes to `docs/` are P0. Existing content under `docs/`,
  including `docs/about/release-notes/`, is tracked for removal —
  see Advocate.
- **Agentic surface features** are product features, not extras:
  - Local and global chat (Ask AI on every page)
  - `llms.txt` and machine-readable markdown views
  - Copy page, View as Markdown, Open in Cloud
  - MCP server integration
  - Algolia-powered search
  - Dashboard for search and chat analytics
- **Versions and redirects.**
  `fern/versions/{latest,main,v25.09,v26.02,v26.04}.yml`, with
  matching directories for `main` and each `vYY.MM` (`latest.yml`
  is redirect-only — no `latest/` directory). Adding a version
  coordinates `fern/docs.yml` redirects and inbound-link impact.
- **No fabricated claims.** Every documented flag, config field,
  classifier name, codec, default, or version pin must trace to
  source. Every snippet must round-trip: `from nemo_curator…`
  imports resolve, CLI lines match argparse,
  `pipeline.add_stage(...)` examples type-check.
- **Cross-page consistency.** Same fact must read identically across
  `fern/`, `README.md`, `CONTRIBUTING.md`, `api-design.md`, cursor
  rules, copilot instructions, and tutorials.
- **Broken-link tooling.** `fern/_fix_broken_links.py` is
  authoritative for Fern. `docs/broken_links_*.json` reflects the
  deprecated Sphinx site — ignore for Fern.
- **Variable substitution.** `fern/substitute_variables.py` rewrites
  `{{ current_release }}` / `<release/>`. Don't hand-pin versions
  where substitution would apply.

## Contract Checklist

When `fern/` changes:

- `fern/docs.yml`, `fern/versions/*.yml` and matching directories
- `fern/AUTODOCS_GUIDE.md`, `fern/README.md`, `fern/components/`,
  `fern/main.css`, `fern/assets/`, `fern/package.json`,
  `fern/fern.config.json`
- `fern/_fix_broken_links.py`, `fern/substitute_variables.py`
- `requirements-docs.txt`
- `.claude/skills/nemo-curator-docs/` — keep aligned
- `.cursor/rules/*.mdc`, `.github/copilot-instructions.md` — any
  product fact shared with `fern/`
- `CHANGELOG.md` and release notes (in `fern/`)

For IA refactors, version cuts, and large content updates, run the
full Content Audit swarm and gate merge on verified P0.

## Doc Autopilot

Three triggers defined in root [AGENTS.md](../AGENTS.md) — merge gate,
periodic re-audit, source-triggered re-audit. **Current state:
manual rollout, automation pending.** No CI gate, scheduled job, or
source-watch wiring exists yet. Wiring these into
`.github/workflows/` is open advocacy below.

Each scoped steward lists its owned doc pages under **Own**. That
list is the audit surface for that steward in autopilot mode.

## Advocate

- **Pin owned doc paths** in every scoped `AGENTS.md`. Most currently
  defer this to "the next docs autopilot pass" — close the gap.
- **Decommission `docs/`** under the 2026-05-20 freeze: confirm Fern
  parity for every migrated page, retire `docs/conf.py`, drop
  `docs/about/release-notes/`, remove or rebase stale redirects.
- **Wire Doc Autopilot triggers into CI**: a `docs-audit-required`
  PR check for the merge gate, a scheduled workflow for periodic
  re-audit, a labels-or-paths trigger for source-triggered re-audits.
- **Programmatic counts** — surface classifier / embedder / codec
  inventories from source so docs don't hand-count.
- **Site-wide grep tool** for the Global Sweep On Accepted P0s rule
  in root AGENTS.md.
- **Health metrics** — track broken-link rate, freshness, owner
  coverage on the cadence in root AGENTS.md Measurement.

## Own

**Content:**

- `fern/` (entire tree); cross-cutting concerns (welcome,
  getting-started, install, glossary, contributor pages,
  release-notes) are this steward's direct audit surface. Scoped
  stewards own their domain pages.
- `requirements-docs.txt`
- Release notes (in `fern/`)
- `CHANGELOG.md` (cross-owned with the implementing area)

**Tests:** any link / lint / structural checks for `fern/` (add a CI
gate if not present — see Advocate).

**Agent artifacts:** `.claude/skills/nemo-curator-docs/` (this
steward owns the skill's accuracy; apply the Docs-First evaluation
gate before expanding).

**CODEOWNERS:** `@NVIDIA-NeMo/docs_team` for both `docs/` and
`fern/`.
