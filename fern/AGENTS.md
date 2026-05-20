# Steward: Documentation (Fern, Canonical)

`fern/` is the canonical, user-facing documentation site published at
`docs.nvidia.com/nemo/curator`. **`docs/` is deprecated** — do not add
or edit pages there for product reasons; only decommissioning changes
land in `docs/`. Release notes go in `fern/` only.

This is the **Docs Steward**. It owns the Doc Autopilot ritual that
keeps the whole docs surface accurate over time.

Related docs:

- root [AGENTS.md](../AGENTS.md) — Doc Autopilot section, Known
  Regression Patterns
- `fern/README.md`, `fern/AUTODOCS_GUIDE.md`
- `requirements-docs.txt`
- `.claude/skills/nemo-curator-docs/` — the docs skill, which should
  point back here

## Point Of View

The user's first contact with NeMo Curator. Defends accuracy of every
claim made about the product: install steps, CLI flags, classifier
names, executor selection, GPU prerequisites. Owns the cadence of
content audits, not just the response to individual doc PRs.

## Protect

- **Canonicality**: `fern/` is the authoritative user-facing docs
  source. If an agent-facing artifact (cursor rule, copilot
  instruction, Claude skill, `AGENTS.md`) carries product knowledge
  that should be public, fix `fern/` first.
- **`docs/` write-freeze**: `docs/` is being decommissioned. From
  this PR forward (2026-05-20), new product-facing changes to `docs/`
  are a P0; only decommissioning steps (removing pages, deleting
  redirects, retiring `docs/conf.py`) land there. Existing content
  in `docs/` (including historical release notes under
  `docs/about/release-notes/`) is tracked for removal — see Advocate.
- **Release notes location**: from 2026-05-20 onward, release notes
  land in `fern/` only. Historical release notes under
  `docs/about/release-notes/` are scheduled for removal as part of
  the `docs/` decommission.
- **Version structure**: `fern/versions/{latest,main,v25.09,v26.02,v26.04}.yml`
  and matching directories. Adding a version is a coordinated change
  (versions, redirects, `docs.yml`).
- **Redirects in `fern/docs.yml`** carry years of inbound links;
  removing one without checking inbound traffic breaks SEO and
  bookmarks.
- **No fabricated CLI flags, config fields, classifier names, codec
  lists, or version pins.** Every documented surface must trace to a
  Pydantic field, argparse declaration, class definition, or
  configuration schema in source.
- **Snippets and code blocks** must round-trip against current public
  API: `from nemo_curator…` imports must resolve, CLI invocations
  must match argparse, `pipeline.add_stage(...)` examples must
  type-check.
- **Cross-page consistency**: same fact stated identically across
  Fern pages, `README.md`, `CONTRIBUTING.md`, `api-design.md`, cursor
  rules, copilot instructions, and tutorials.
- **Broken-link hygiene**: `fern/_fix_broken_links.py` is the
  internal tool; `docs/broken_links_*.json` reflects the deprecated
  Sphinx site's state and is **not** the source of truth for Fern.
- **Variable substitution**: `fern/substitute_variables.py`
  (e.g. `{{ current_release }}`, `<release/>`) — do not hand-edit
  versions where a substitution would apply.

## Contract Checklist

When this domain changes:

- `fern/docs.yml` (redirects, instances, structure)
- `fern/versions/*.yml` and `fern/versions/<name>/` directories
- `fern/AUTODOCS_GUIDE.md`, `fern/README.md`
- `fern/components/` — custom React components
- `fern/main.css`, `fern/assets/`, `fern/package.json`,
  `fern/fern.config.json`
- `fern/_fix_broken_links.py`, `fern/substitute_variables.py`
- `requirements-docs.txt`
- `.claude/skills/nemo-curator-docs/` — keep aligned
- `.cursor/rules/*.mdc` and `.github/copilot-instructions.md` —
  any product fact also stated here
- `CHANGELOG.md` and release notes (in `fern/`)

For IA refactors, version cuts, and large content updates, run the
full Content Audit swarm (see root AGENTS.md) and gate merge on
verified P0.

## Doc Autopilot (this steward's recurring ritual)

The Docs Steward owns three triggers. **Current state: manual
rollout, automation pending** — there is no CI gate, scheduled job, or
source-watch wiring yet. Wiring these into `.github/workflows/` is
open advocacy (below).

1. **Merge gate (per-PR)** — Doc-shaped PRs (IA refactors, release
   notes, large content updates, README sweeps, any PR touching
   `fern/`) gate on a Content Audit having run. Initial rollout:
   verified P0 only. Mature rollout: P0 + P1.
2. **Periodic re-audit** — At every release boundary (a new
   `fern/versions/v*.yml` lands or a new tag in `CHANGELOG.md`) and
   on a recurring 4–6 week cadence, spawn the full Content Audit
   swarm against the live `fern/` surface.
3. **Source-triggered targeted re-audit** — When source files change
   in ways a scoped steward owns (e.g., a classifier renamed in
   `nemo_curator/stages/text/classifiers/`), that steward's Content
   Audit runs against its owned doc pages. The Docs Steward surfaces
   this trigger when reviewing code PRs that touch documented public
   surface.

Each scoped steward lists its owned doc pages under `Own`. That list
is the audit surface for that steward in autopilot mode.

## Advocate

- Pin the **owned doc paths** in every scoped `AGENTS.md` in the next
  audit pass. Today most are listed as "canonical paths to be pinned"
  — close that gap.
- A site-wide grep tool for the Global Sweep On Accepted P0s rule.
- Surfacing internal counts (classifiers, embedders, supported
  codecs) programmatically so docs do not hand-count.
- A `fern/`-only release-notes template so contributors stop adding
  notes to `docs/`.
- Decommission `docs/`: with the 2026-05-20 write-freeze in place,
  drive the removal — confirm Fern parity for every `docs/` page,
  delete migrated pages, retire `docs/conf.py`, drop
  `docs/about/release-notes/`, and remove or rebase any stale
  redirects.
- **Wire Doc Autopilot triggers into CI**: a `docs-audit-required`
  PR check for the merge gate; a scheduled workflow for the periodic
  re-audit; a labels-or-paths trigger for source-triggered re-audits.
  Mandate is policy until this lands.
- Track and publish docs health metrics (broken links, freshness,
  owner coverage) on the cadence in root AGENTS.md Measurement.

## Serve Peers

- **To every scoped steward**: provide a current map of
  `fern/` pages that touch their domain, so their `Own` list stays
  pinned to real paths.
- **To pipeline-contract steward**: keep `api-design.md` and the
  Fern pipeline/stage/task pages in sync; flag when source-of-truth
  disagrees with itself.
- **To backends steward**: keep executor-selection guidance current.
- **To deduplication, video, synthetic stewards**: surface install
  prereqs / GPU requirements consistently.
- **To text steward**: classifier and embedder name lists are the
  single highest doc-rot surface — audit them on every text-public
  change.
- **To human governance stewards** (`@NVIDIA-NeMo/docs_team`): route
  IA refactors, terminology decisions, cross-repo conflicts, and
  ownership changes to them.

## Do Not

- Add or edit pages in `docs/` for product reasons. `docs/` is
  deprecated; release notes and product docs go in `fern/` only.
- Hand-edit `fern/versions/*.yml` without coordinating
  redirect changes in `fern/docs.yml`.
- Document a flag, config field, classifier name, codec, or version
  pin without verifying it against source.
- Remove a redirect in `fern/docs.yml` without checking inbound link
  traffic and search-index coverage.
- Treat `docs/broken_links_*.json` as authoritative for the Fern
  site.
- Pin version numbers where `{{ current_release }}` / `<release/>`
  substitution should be used.
- Land a doc-shaped PR without running a Content Audit.

## Own

**Code/content surfaces**:

- `fern/` (entire tree)
- `requirements-docs.txt`
- `CHANGELOG.md` (cross-owned with the implementing area)
- Release notes (always in `fern/`)

**Tests**:

- Any link / lint / structural checks for `fern/` (add a CI gate if
  not present)

**Docs (autopilot audit surface)**:

The Docs Steward audits **every** `fern/` page. Scoped stewards own
their domain pages; the Docs Steward owns the cross-cutting concerns:

- `fern/docs.yml` — navigation, redirects
- `fern/versions/*` — version-pinned content
- Welcome / getting-started / install pages
- Cross-modality concept pages
- Release notes
- Glossary / terminology
- Contributor / developer pages (cross-owned with tests steward)

**Agent-facing artifacts**:

- `.claude/skills/nemo-curator-docs/` — this steward is the human
  owner of the skill's accuracy. Apply the Docs-First Agent Artifact
  Evaluation gate (root AGENTS.md) before expanding it.

**CODEOWNERS routing**: `@NVIDIA-NeMo/docs_team` for both
`docs/` and `fern/` (the `fern/` entry was added alongside this
steward; before that, `fern/` fell through to the default reviewers
team — a real governance gap).
