# Steward Manifest Schema

`.stewards/manifest.toml` is the source of truth for generated `AGENTS.md`
maps. Stable repository knowledge belongs in the manifest; generated maps stay
small enough for ordinary task context.

Top-level fields define the network, active-context budget, code-domain
coverage roots, project pillars, progressive search policy, operating rules,
stop conditions, and done criteria.

All manifest paths are repository-relative. Absolute paths and `..` traversal
are rejected before projection, and generated maps not represented by the
manifest are rejected as orphans. A coverage exemption must name an existing
domain and give a non-empty inheritance reason.

Each `[check.<id>]` registers an exact command and an existing proof location.
Optional `proof_contains` text prevents a check from being wired to an
unrelated file merely because the path exists.

Each `[[steward]]` owns one generated map and may declare code, test, and docs
paths, guardrails, and typed graph edges to other stewards.

Each `[[invariant]]` is `machine`, `manual`, or explicit `none`:

- `machine` names a registered check through `enforced_by`; every check has a
  proof location and `proof_contains` anchor.
- `manual` names an evidence file and stable text anchor.
- `none` records visible verification debt rather than overstating coverage.

Invariant severity is one of `P0`, `P1`, `P2`, or `P3`. Severity describes
demonstrated contract impact, not how many reviewers or stewards agree.

`[judgment.<steward>]` contains non-enforceable advocacy, refusal patterns, and
the peers a steward serves.

Generated maps render exact machine-check commands. Ordinary agents should not
open this manifest or the protocol merely to translate check identifiers.

Validate with Python 3.11 or newer:

```bash
python .stewards/project.py --check
python .stewards/verify.py --coverage
```
