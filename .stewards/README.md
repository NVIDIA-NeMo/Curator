# Steward Authoring

The `AGENTS.md` files in this repository are compact generated views. Core
engineers and domain owners still author the guidance: they edit the source
layer named at the top of the generated map.

## Update guidance for a domain

1. Open the relevant generated `AGENTS.md`.
2. Follow its source-layer link, such as
   `.stewards/layers/backends.toml`.
3. Make the smallest evidence-backed change in that layer.
4. Regenerate and verify:

   ```bash
   python .stewards/project.py
   python .stewards/verify.py --coverage
   ```

5. Commit the source layer and generated map together. `CODEOWNERS` requests
   review from the engineers who own that guidance.

Use the layer for durable lessons such as a recurring agent assumption, an
important preferred pattern, a known exception, or a domain contract that is
not obvious from local code. Do not add guidance that an agent can infer
reliably from the source.

## Choose the right record

- Add a `guardrails` item when a recurring mistake needs a concise passive
  warning.
- Add `advocate`, `do_not`, or `serves` judgment when the guidance expresses a
  domain-owner preference or review viewpoint.
- Add an `invariant` when a stable contract has a severity and an exact machine
  check, evidence anchor, or explicit verification gap.
- Change the central manifest when the network, check registry, coverage
  boundary, or context budget changes.

## Understand layer resolution

`.stewards/manifest.toml` lists source layers in deterministic precedence
order. The projector resolves them like a small configuration chart:

```text
repository base
      ↓
domain layer
      ↓
optional overlay
      ↓
generated AGENTS.md
```

Lists merge without exact duplicates. Later scalar values win. A layer cannot
move an existing steward map to another path. Replacing an invariant with the
same stable ID requires `override = true`, which makes the replacement
intentional and reviewable.

Show the effective map and source chain for one domain:

```bash
python .stewards/project.py --explain backends
```

## Add a scoped map

To spawn an `AGENTS.md` for a new ownership boundary:

1. Add a layer file under `.stewards/layers/`.
2. Register the file in `layer_sources` in `.stewards/manifest.toml`.
3. Declare a new `[[steward]]` with a unique ID and generated map path.
4. Add at least one invariant for the steward.
5. Add an exact `CODEOWNERS` entry for the source layer.
6. Run the projector and verifier.

The projector creates the declared map. The verifier rejects unsafe paths,
duplicate scopes, missing ownership, uncovered code domains, stale output,
orphan generated maps, and active context that exceeds the configured budget.

Do not create a scoped map only because a CODEOWNERS boundary exists. Add one
when work in that path needs materially different passive guidance.

## Add an overlay

An overlay can add narrower guidance without copying its base layer:

```toml
[layer]
id = "text-classifiers"
target = "text"
kind = "overlay"
owners = ["@sarahyurick"]

[[steward]]
id = "text"
guardrails = [
  "Prefer the existing classifier stages before introducing a parallel abstraction.",
]
```

Register the overlay after its domain layer. Its content merges into the same
generated map, and the generated header lists both sources and their owners.
