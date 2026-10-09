# NeMo Curator — Fern Docs

This directory holds the Fern MDX source for the NeMo Curator documentation site at **[docs.nvidia.com/nemo/curator](https://docs.nvidia.com/nemo/curator)**.

All new pages and edits land here. The Sphinx tree that used to live under `../docs/` has been retired and replaced with a pointer README.

## Quick links

| What | Where |
|---|---|
| Published site | https://docs.nvidia.com/nemo/curator |
| Fern dashboard | https://dashboard.buildwithfern.com (NVIDIA org) |
| Autodocs (libraries) guide | [`./AUTODOCS_GUIDE.md`](./AUTODOCS_GUIDE.md) |
| CI workflows | [`../.github/workflows/fern-docs-*.yml`](../.github/workflows/) and [`publish-fern-docs.yml`](../.github/workflows/publish-fern-docs.yml) |

## Quickstart

First time on this machine (run from the repo root):

```bash
# 1. Install the Fern CLI (one-time)
npm install -g fern-api
# or use it ad-hoc via:  npx -y fern-api@latest <subcommand>

# 2. Sign in to the Fern dashboard (https://dashboard.buildwithfern.com),
#    then authenticate the CLI:
fern login

# 3. Validate config + MDX
cd fern && fern check

# 4. Start the local dev server
fern docs dev          # http://localhost:3000
```

**Dashboard sign-in is load-bearing.** Without it, the CLI's `fern login` flow alone is *not* enough — Fern's library-generation endpoint will reject you with `HTTP 403: User does not belong to organization`. Sign in to the dashboard once so your account record exists in Fern's user DB, then re-run `fern login`.

### Fern CLI + docs reference

| Resource | Link |
|---|---|
| Fern docs (overview, writing, configuration) | https://buildwithfern.com/learn/docs/getting-started/overview |
| Fern CLI reference | https://buildwithfern.com/learn/cli-api-reference/cli-reference/overview |
| MDX components (Cards, Callouts, Tabs, …) | https://buildwithfern.com/learn/docs/content/components/overview |
| Frontmatter fields | https://buildwithfern.com/learn/docs/content/frontmatter |
| Versioning | https://buildwithfern.com/learn/docs/configuration/versions |
| Redirects | https://buildwithfern.com/learn/docs/seo/redirects |
| `libraries:` (Python autodoc) | https://buildwithfern.com/learn/docs |

## Layout

```
fern/
├── fern.config.json          # Fern CLI org slug + version pin
├── docs.yml                  # Site config: instances, versions, redirects, libraries, theme
├── main.css                  # NVIDIA-green theme overrides
├── assets/                   # Logos, shared SVGs, page images
├── components/               # CurrentRelease, CustomFooter (TSX)
├── versions/
│   ├── main.yml              # Nav for the bleeding-edge train — paths point at ./main/pages/
│   ├── main/pages/           # Bleeding-edge MDX (every PR lands here; published at /main/...)
│   ├── v26.09.yml            # 26.09 docs-train snapshot
│   ├── v26.09/pages/         # Snapshot of main at the 26.09 docs cut
│   ├── v26.07.yml            # Frozen 26.07 docs train
│   ├── v26.07/pages/         # Frozen 26.07 content
│   ├── v26.04.yml            # Frozen 26.04 docs train
│   ├── v26.04/pages/         # Frozen 26.04 content
│   ├── v26.02.yml            # Frozen 26.02 docs train
│   ├── v26.02/pages/         # Frozen 26.02 content
│   ├── v25.09.yml            # Frozen 25.09 GA snapshot
│   ├── v25.09/pages/         # Frozen 25.09 content
│   └── latest.yml            # Symlink → v26.09.yml (current docs alias; retargeted at next cut)
├── substitute_variables.py   # CI/local: substitute {{ variables }} in MDX before generate
├── AUTODOCS_GUIDE.md         # How the `libraries:` block builds the Python API reference
└── product-docs/             # GENERATED Python API reference (gitignored)
```

```
File path                                              Published URL
─────────────────────────────────────────────────────  ─────────────────────────────────────────────────
fern/versions/main/pages/get-started/index.mdx         docs.nvidia.com/nemo/curator/main/get-started
fern/versions/v26.09/pages/get-started/index.mdx       docs.nvidia.com/nemo/curator/v26.09/get-started
                                                       docs.nvidia.com/nemo/curator/latest/get-started   (latest aliases v26.09)
fern/versions/v26.07/pages/get-started/index.mdx       docs.nvidia.com/nemo/curator/v26.07/get-started
fern/versions/v26.04/pages/get-started/index.mdx       docs.nvidia.com/nemo/curator/v26.04/get-started
fern/versions/v26.02/pages/get-started/index.mdx       docs.nvidia.com/nemo/curator/v26.02/get-started
fern/versions/v25.09/pages/get-started/index.mdx       docs.nvidia.com/nemo/curator/v25.09/get-started
```

**`main/pages/` is the bleeding-edge tree** — every PR lands here and publishes under the `main` slug (`availability: beta`, shown as `Main · preview` in the picker). **`v26.09/pages/` is the current docs-train snapshot**, cut from `main/pages/`; `main/` remains the preview source for ongoing work. Earlier version trees are preserved as historical documentation. `latest.yml` is a symlink to the current docs train's nav (today: `v26.09.yml`), so `/latest/...` URLs serve the current train. At the next cut, snapshot `main/` to a new `vXX.YY/` and retarget the symlink.

`display-name` in `docs.yml` pairs the **NeMo calendar docs train** (e.g. `26.09`) with its **planned release tag** (e.g. `v1.4.0`) so the version picker reflects the release mapping. Check the release notes and GitHub release information before cutting a train; a docs cut does not itself publish a package release. The `main` entry stays unpinned (`Main · preview`) because it tracks unreleased work.

## Local development

From the `fern/` directory:

```bash
fern check                  # config + MDX validation
fern docs dev               # http://localhost:3000
fern generate --docs        # static build
```

For prose-only iteration, `fern docs dev` is enough. Library reference (`product-docs/`) is regenerated by Fern on `dev`/`generate` from the `libraries:` block in `docs.yml`. See [`./AUTODOCS_GUIDE.md`](./AUTODOCS_GUIDE.md) for how that works.

### Library reference without Fern login

Deploy previews and publication use the `nemo-curator` library (`input.git`) and require `fern login` or `FERN_TOKEN`. PR configuration checks are token-free and do not generate authenticated autodocs. For local iteration on the Python API reference without Fern authentication, run:

```bash
npm run generate:library:local   # runs scripts/generate-library-local.sh; requires Docker
```

The script temporarily injects a path-input `nemo-curator-local` entry into `docs.yml`, runs `fern docs md generate --local --library nemo-curator-local`, and restores `docs.yml` afterward. The entry can't live in the committed `docs.yml`: `fern docs dev` and `fern generate --docs` reject any `input.path` library ("'path' input which is not yet supported") and render a blank page, while `--local` generation only accepts `path` inputs. Both flows write to the same output folder (`product-docs/nemo-curator/Full-Library-Reference`), so previews and publish keep working unchanged. Requires Docker (Fern pulls `fernapi/fern-python-library-docs-parser` on first run).

### Variable substitution

CI runs `substitute_variables.py` before `fern generate` so `{{ container_version }}` and other tokens in MDX are replaced. To run locally (from repo root):

```bash
python fern/substitute_variables.py versions/main   --version XX.YY   # preview — pass the next planned train
python fern/substitute_variables.py versions/v26.09 --version 26.09
python fern/substitute_variables.py versions/v26.07 --version 26.07
python fern/substitute_variables.py versions/v26.04 --version 26.04
python fern/substitute_variables.py versions/v26.02 --version 26.02
python fern/substitute_variables.py versions/v25.09 --version 25.09
```

Variables are defined at the top of `substitute_variables.py` (e.g. `{{ product_name }}`, `{{ github_repo }}`, `{{ container_version }}`). For `main/`, pass whatever release is *next on the train* — pages there should display the version they will ship as, not literal `main`.

Document source installation or container builds from the repository Dockerfile in `main` and future version snapshots. NVIDIA no longer publishes NGC images starting with 26.09, so release numbers no longer identify container tags. Keep `{{ container_version }}` in archived pages that document historical tags and use variable substitution.

## Authoring conventions

### Frontmatter

```yaml
---
title: "<Page Title>"        # required — used by Fern as the page title and breadcrumb
description: ""              # required (may be empty string) — SEO
position: 1                  # optional — orders auto-discovered pages within a folder
---
```

The MDX body should still open with `# <Page Title>` matching the frontmatter title. Folders using `title-source: frontmatter` in the version YAML pull the nav label from `title:`.

### Components

| Component | Purpose |
|---|---|
| `<CurrentRelease />` | Inline current-release badge, sourced from version variables |
| `<CustomFooter />` | Wired in `docs.yml` `footer:`; **required** for NVIDIA legal/privacy compliance |

Standard Fern components are also available — `<Note>`, `<Tip>`, `<Info>`, `<Warning>`, `<Cards>` / `<Card>`, `<Badge>`, `<Tabs>`, etc. Don't use GitHub `> [!NOTE]` syntax — it does not render in MDX.

### Internal links

Use **version-prefixed paths** matching the slug of the tree the page lives in:

```mdx
[Get started](/v26.09/get-started)        // links inside versions/v26.09/pages/
[Get started](/v26.07/get-started)        // links inside versions/v26.07/pages/
```

Cross-version links (e.g. from a `v26.09/` page to a `v26.07/` page) trigger broken-link warnings in `fern docs dev`; those are **false positives** — Fern's local validator does not resolve cross-version slugs from `docs.yml`. The published site renders them correctly.

### Cross-repo references (yaml configs, source files)

Repository source paths like `nemo_curator/...` are not part of the docs site. Link to them as **absolute GitHub URLs**:

```mdx
[backends/ray_data](https://github.com/NVIDIA-NeMo/Curator/tree/main/nemo_curator/backends/ray_data)
```

## Versioning

`docs.yml` `versions:` lists the latest alias, the moving preview, and each retained stable docs train:

| display-name | slug | availability | path |
|---|---|---|---|
| `Latest · v1.4.0 (26.09)` | `latest` | `stable` | `./versions/latest.yml` (symlink → `v26.09.yml`) |
| `Main · preview` | `main` | `beta` | `./versions/main.yml` |
| `26.09 · v1.4.0` | `v26.09` | `stable` | `./versions/v26.09.yml` |
| `26.07 · v1.3.0` | `v26.07` | `stable` | `./versions/v26.07.yml` |
| `26.04 · v1.2.0` | `v26.04` | `stable` | `./versions/v26.04.yml` |
| `26.02 · v1.1.0` | `v26.02` | `stable` | `./versions/v26.02.yml` |
| `25.09 · v1.0.0` | `v25.09` | `stable` | `./versions/v25.09.yml` |

When the next docs train is cut (replace `XX.YY` and `vX.Y.Z` with the values confirmed for that release):

1. Copy `versions/main/pages/` to `versions/vXX.YY/pages/` to snapshot the current preview source. Preserve page bytes and placeholders.
2. Copy `versions/main.yml` to `versions/vXX.YY.yml`, then rewrite only `./main/` path prefixes to `./vXX.YY/`.
3. Retarget the latest alias symlink: `cd versions && ln -sfn vXX.YY.yml latest.yml`.
4. Add the new stable train to `docs.yml` `versions:` and update the Latest display name to `Latest · vX.Y.Z (XX.YY)`. Keep `Main · preview` and retained historical trains.
5. Add the calendar-path, `index.html`, and `.html` redirects for `/XX.YY` in `docs.yml` before the generic catch-all redirects.
6. Mirror only existing secret-baseline findings for the copied pages, preserving each finding's type, hash, verification state, and line number.
7. Verify the snapshot matches `main/pages/`, the nav differs only in its version path prefix, and older version trees remain unchanged.

## CI and publishing

| Workflow | Trigger | Purpose |
|---|---|---|
| `fern-docs-ci.yml` | PR touching Fern docs or preview/check code | Token-free `fern check`; no staging regression tests or authenticated autodocs generation |
| `fern-docs-preview.yml` | Push to an approved upstream `pull-request/<number>` mirror | Verify the current PR head, generate a preview with trusted configuration and pinned tooling, and post a comment; remove stale preview comments when docs changes are reverted |
| `publish-fern-docs.yml` | push of `docs/v*` tag, or manual dispatch | Publish to docs.nvidia.com/nemo/curator |

Required org secret: **`DOCS_FERN_TOKEN`** (issued via `fern token` on a privileged dashboard account).

PRs that touch `fern/**` get a preview after their current head is approved and mirrored upstream, including vetted fork PRs. Direct fork runs do not receive the publishing token. This workflow has no `PUBLISH_FERN` feature-flag gate.

Preview navigation, components, organization and tooling come from `main`; PRs supply only documentation content and assets. Existing unchanged trusted symlinks (such as `versions/latest.yml`) are allowed; new or retargeted PR symlinks are rejected. Explicit page/asset deletions relative to the PR merge base are applied, while pages added only on newer `main` remain available. Navigation changes are not previewed until they land on `main`; deleting a still-referenced page can therefore fail the build rather than silently showing its old content.

Fern CLI **5.149.0** is pinned in workflow code with `FERN_NO_VERSION_REDIRECTION=true`. Preview token access is limited to library generation and publication. The comment URL must match the exact expected Fern preview host; no authenticated page-link request is made. Reverting all docs changes removes existing bot preview comments without publishing or accessing the Fern token.

### Publishing to production

```bash
git tag docs/v1.2.0
git push origin docs/v1.2.0
```

Or trigger manually from the **Actions** tab → **Publish Fern Docs** → **Run workflow**.

## Commits

DCO sign-off is required:

```bash
git commit -s -m "docs: <add|update|remove> <page-title>"
```

PR titles follow Conventional Commits (e.g. `docs(fern): add rollout collection guide`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `HTTP 403: User does not belong to organization` on library generation | Sign in to https://dashboard.buildwithfern.com first, then re-run `fern login`. Or use `npm run generate:library:local` (no Fern auth; requires Docker). |
| `Folder not found: ./product-docs/...` in `fern docs dev` | Library generation hasn't run; run `npm run generate:library` (or `generate:library:local`) first |
| `Library '...' uses 'path' input which is not yet supported` + blank page in `fern docs dev` | A path-input library entry is committed in `docs.yml` — remove it; local generation injects it temporarily via `scripts/generate-library-local.sh` |
| `fern check` YAML error | 2-space indent; `- page:` inside `contents:`; `path:` is relative to the version YAML |
| Page 404 in preview | `slug:` missing/duplicated in the same section, or `position:` collision in an auto-discovered folder |
| Broken-link warning for cross-version path | False positive in `fern docs dev`; the published site resolves it correctly |
| `JSX expressions must have one parent element` | Wrap multi-element MDX content in `<>...</>` or a `<div>` |
| Old `/index.html` or `/foo.html` URL breaks | Add a `redirects:` entry in `docs.yml` (catch-alls for `:path*/index.html` and `:path*.html` already exist) |
| `{{ variable }}` shows literally on the published page | Run `python fern/substitute_variables.py versions/vXX.YY --version XX.YY` before `fern generate`, or check the variable is registered in `DEFAULT_VARIABLES` |
| Library reference missing or stale | Check `libraries:` block in `docs.yml` matches the package source path; see [`./AUTODOCS_GUIDE.md`](./AUTODOCS_GUIDE.md) |

## Reference

- [Fern docs (upstream)](https://buildwithfern.com/learn/docs/getting-started/overview)
- [NeMo Curator source](https://github.com/NVIDIA-NeMo/Curator)
