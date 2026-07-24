# Steward Manifest Schema

`.stewards/manifest.toml` is the control plane for generated `AGENTS.md` maps.
Owner-editable source layers hold repository and domain guidance. The projector
resolves the ordered layers and emits compact maps for ordinary agent context.

For the domain-owner workflow, see `.stewards/README.md`.

## Manifest

Top-level fields define:

- the network name and on-demand review protocol;
- the active-context byte budget;
- code-domain coverage roots and reasoned exemptions;
- the ordered `layer_sources`;
- exact machine checks and their proof locations.

Layer paths are repository-relative. Absolute paths and `..` traversal are
rejected before a layer is opened.

## Layer metadata

Every source layer starts with:

```toml
[layer]
id = "backends"
target = "backends"
kind = "domain"
owners = ["@oyilmaz-nvidia"]
```

- `id` uniquely identifies the layer.
- `target` identifies the steward map receiving its content.
- `kind` is `base`, `domain`, or `overlay`.
- `owners` lists GitHub users or teams that author the guidance.

Each source layer requires an exact matching entry in `.github/CODEOWNERS`.
The verifier rejects missing owners, invalid handles, and mismatched ownership.

The first `base` or `domain` layer for a target declares its `[[steward]]`.
An `overlay` may target only a steward declared by an earlier layer.

## Resolution

Layers resolve in `layer_sources` order:

- lists append and discard exact duplicates;
- mappings merge recursively;
- later scalar values take precedence;
- an overlay cannot change an existing steward's generated map path;
- duplicate invariant IDs fail unless the later invariant declares
  `override = true`.

The generated map header records all contributing layers and owners. Use
`python .stewards/project.py --explain <steward>` to inspect the effective
root-to-scope map chain.

## Steward

Each `[[steward]]` owns one generated map and may declare:

- `path`: the generated `AGENTS.md` path;
- `point_of_view`: the domain contract in one sentence;
- `guardrails`: compact recurring warnings;
- `owns`: code, test, and documentation paths;
- `edges`: typed relationships to other stewards.

Adding a new steward layer and registering it in `layer_sources` causes the
projector to create the declared map. Generated maps not represented by the
resolved layers are rejected as orphans.

## Invariant

Each `[[invariant]]` is `machine`, `manual`, or explicit `none`:

- `machine` names a registered check through `enforced_by`; every check has a
  proof location and `proof_contains` anchor.
- `manual` names an evidence file and stable text anchor.
- `none` records visible verification debt rather than overstating coverage.

Invariant severity is `P0`, `P1`, `P2`, or `P3`. Severity describes
demonstrated contract impact, not how many reviewers or stewards agree.

## Judgment

`[judgment.<steward>]` contains non-enforceable owner guidance:

- `advocate`: preferred patterns or review priorities;
- `do_not`: recurring mistakes and unsupported assumptions;
- `serves`: the users and peer domains affected by the guidance.

## Checks and coverage

Each `[check.<id>]` registers an exact command, an existing proof location, and
stable `proof_contains` text. A coverage exemption must name an existing domain
and give a non-empty inheritance reason.

Generated maps render exact machine-check commands. Ordinary agents should not
open source layers or the protocol merely to translate a check identifier.

Validate with Python 3.11 or newer:

```bash
python .stewards/project.py --check
python .stewards/verify.py --coverage
```
