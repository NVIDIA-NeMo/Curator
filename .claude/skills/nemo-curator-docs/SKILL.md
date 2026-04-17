---
name: nemo-curator-docs
description: Maintain the NeMo Curator Fern docs site — add, update, move, or remove pages under fern/. Use for any documentation changes.
---

# NeMo Curator Docs Maintenance

Unified skill for adding, updating, moving, and removing pages on the NeMo Curator Fern documentation site.

## Scope Rule

**ALL docs edits happen under `fern/`.** The legacy `docs/` directory is deprecated — do not add or move content into it. Release notes, migration guides, and every new page belong under `fern/`.

## Layout at a Glance

```
fern/
├── fern.config.json          # Minimal Fern config (org + CLI version)
├── docs.yml                  # Site config: versions, tabs, redirects, libraries
├── versions/
│   ├── latest.yml            # Symlink → v26.02.yml (do not edit directly)
│   ├── v26.02.yml            # Nav tree for current train
│   ├── v26.02/pages/         # MDX content for current train
│   ├── v25.09.yml
│   └── v25.09/pages/
├── components/               # Custom TSX components (footer, etc.)
├── assets/                   # Images, SVGs, favicon
├── substitute_variables.py   # CI: resolves {{ variables }} in MDX
└── AUTODOCS_GUIDE.md         # Library reference generation guide
```

**Current train:** `v26.02`. Default all new pages there unless the user specifies a version.

```
File system                              Published URL
───────────────────────────────────────  ────────────────────────────────────────
fern/versions/v26.02/pages/              docs.nvidia.com/nemo/curator/latest/
  └─ get-started/text.mdx                  └─ get-started/text
fern/versions/v26.02.yml ── nav for ──┐  docs.nvidia.com/nemo/curator/v26.02/
fern/versions/latest.yml ─ symlink ───┘    └─ get-started/text
fern/versions/v25.09/pages/              docs.nvidia.com/nemo/curator/v25.09/
  └─ get-started/text.mdx                  └─ get-started/text
```

## Operations

### Add a Page

1. Gather: page title, target section, filename (kebab-case `.mdx`), subdirectory under `fern/versions/v26.02/pages/`.
2. Create `fern/versions/v26.02/pages/<subdirectory>/<filename>.mdx`:

```mdx
---
description: "One-line SEO description"
categories: ["<category>"]
tags: ["<tag-1>", "<tag-2>"]
personas: ["<persona>"]
difficulty: "beginner"      # beginner | intermediate | advanced
content_type: "tutorial"     # tutorial | how-to | reference | concept | index
modality: "text-only"        # text-only | image-only | video-only | audio-only | universal
---

# <Page Title>

<content>
```

3. Add a nav entry in `fern/versions/v26.02.yml` under the correct section:

```yaml
- page: <Page Title>
  path: ./v26.02/pages/<subdirectory>/<filename>.mdx
  slug: <filename>
```

4. If this also applies to `latest`, no action needed — `latest.yml` is a symlink to `v26.02.yml`.

### Update a Page

1. Locate by path, title, or keyword (`grep -rn` in `fern/versions/v26.02/pages/`).
2. **Content only** — edit the MDX directly.
3. **Title change** — update the frontmatter and the `- page:` name in `fern/versions/v26.02.yml`.
4. **Section move** — `git mv` the file, update its `path:` in the nav, and fix all incoming links.
5. **Slug change** — update `slug:` in the nav and add a redirect in `fern/docs.yml` so old URLs keep working.

### Remove a Page

1. Find incoming links: `grep -r "<filename>" fern/versions/v26.02/pages/ --include="*.mdx"`.
2. `git rm fern/versions/v26.02/pages/<subdirectory>/<filename>.mdx`.
3. Remove the `- page:` block from `fern/versions/v26.02.yml`. If it was the last page in a section, remove the `- section:` block.
4. Fix or remove all incoming links found in step 1.
5. Add a redirect in `fern/docs.yml` if the URL was public.

### Back-port to an Older Version

Only when explicitly asked. Repeat the operation in the corresponding `fern/versions/vXX.YY/` tree and `vXX.YY.yml` nav. MDX content often diverges between trains — do not blindly copy.

### Worked Example: Adding a Page

Request: *"Add a how-to for benchmarking text pipelines under Curate Text."*

1. Create `fern/versions/v26.02/pages/curate-text/benchmarking.mdx`:

   ```mdx
   ---
   description: "Benchmark text curation pipelines and interpret throughput and memory metrics"
   categories: ["how-to"]
   tags: ["text-curation", "benchmarking", "performance"]
   personas: ["mle-focused"]
   difficulty: "intermediate"
   content_type: "how-to"
   modality: "text-only"
   ---

   # Benchmark Text Pipelines

   <content>
   ```

2. Add nav entry in `fern/versions/v26.02.yml` under the existing `Curate Text` section:

   ```yaml
   - page: Benchmark Text Pipelines
     path: ./v26.02/pages/curate-text/benchmarking.mdx
     slug: benchmarking
   ```

3. `cd fern && fern check` then `fern docs dev` and verify the page renders at `/curate-text/benchmarking`.

### Worked Example: Renaming a Slug (with Redirect)

Request: *"Rename `/curate-text/benchmarking` to `/curate-text/performance`."*

1. Update `slug:` in `fern/versions/v26.02.yml`: `slug: performance`.
2. (Optional) `git mv` the MDX file if you want the filename to match the slug.
3. Add a redirect to `fern/docs.yml` so old links keep working:

   ```yaml
   redirects:
     - source: "/nemo/curator/latest/curate-text/benchmarking"
       destination: "/nemo/curator/latest/curate-text/performance"
     - source: "/nemo/curator/v26.02/curate-text/benchmarking"
       destination: "/nemo/curator/v26.02/curate-text/performance"
   ```

4. `grep -rn "/curate-text/benchmarking" fern/versions/v26.02/pages/` and update any incoming links.

---

## Content Guidelines

NeMo Curator uses **Fern-native MDX components directly** (unlike Dynamo, which converts GitHub callouts in CI). Do not use `> [!NOTE]` syntax — it will not render.

| Purpose | Component |
|---|---|
| Neutral aside | `<Note>...</Note>` |
| Helpful tip | `<Tip>...</Tip>` |
| Informational callout | `<Info>...</Info>` |
| Warning | `<Warning>...</Warning>` |
| Error / danger | `<Error>...</Error>` |
| Card grid on index pages | `<Cards>` with `<Card title="..." href="...">` children |

Images live in `fern/assets/` (shared) or `fern/versions/vXX.YY/pages/_images/` (version-scoped). Reference with root-relative paths.

Component examples:

```mdx
<Tip>
If `uv` is not installed, see the [Installation Guide](/admin/installation).
</Tip>

<Warning>
GPU-accelerated dedup requires CUDA {{ recommended_cuda }} or later.
</Warning>

<Cards>
  <Card title="Text Curation" href="/get-started/text">
    Set up and run text curation workflows.
  </Card>
  <Card title="Image Curation" href="/get-started/image">
    Set up and run image curation workflows.
  </Card>
</Cards>
```

## Frontmatter Fields

Required: `description`.
Optional but strongly preferred: `categories`, `tags`, `personas`, `difficulty`, `content_type`, `modality`. Existing pages in the same section are the best reference for valid values.

`title` is taken from the `- page:` entry in the nav file; the MDX file itself uses an `# H1` heading matching the page name.

## Variable Substitution

Tokens like `{{ product_name }}`, `{{ container_version }}`, `{{ current_release }}`, `{{ github_repo }}`, `{{ min_python_version }}` are resolved by `fern/substitute_variables.py` at CI time. Use them instead of hard-coding versions or URLs. Canonical list in `DEFAULT_VARIABLES` at the top of that file.

Example in MDX:

```mdx
Install {{ product_name }} {{ current_release }} from {{ github_repo }}.
Requires Python {{ min_python_version }}+ and CUDA {{ recommended_cuda }}.
```

After substitution at CI time:

```
Install NeMo Curator 25.09 from https://github.com/NVIDIA-NeMo/Curator.
Requires Python 3.10+ and CUDA 12.0+.
```

To preview substitution locally:

```bash
python fern/substitute_variables.py versions/v26.02 --version 26.02 --dry-run
```

## Validate

```bash
cd fern
fern check                   # YAML + frontmatter validation
fern docs broken-links       # link check
fern docs dev                # localhost:3000 hot-reload preview
```

`fern check` must pass before commit. Broken-link check can be deferred but must pass in CI.

## Commit & Preview

```bash
git add fern/
git commit -s -m "docs: <add|update|remove> <page-title>"
```

PRs that touch `fern/**` get an automatic Fern preview URL posted as a comment by `.github/workflows/fern-docs-preview.yml`. No manual step needed.

```
                    ┌─ fern-docs-ci.yml         → fern check + autodocs
PR (touches fern/) ─┼─ fern-docs-preview.yml    → preview build
                    └─ fern-docs-preview-*.yml  → 🌿 preview URL comment

Merge to main      → NO publish. Site is unchanged.

Tag push (docs/v*) → publish-fern-docs.yml      → docs.nvidia.com/nemo/curator
```

## Publishing to Production

**Merging to `main` does NOT publish.** Production only updates when a tag matching `docs/v*` is pushed (or the workflow is manually dispatched from the **Actions** tab). Do not push tags unless the user asks.

Tag must be `docs/v<MAJOR>.<MINOR>.<PATCH>` — the `docs/v` prefix is required by the workflow trigger and the semver suffix should match the docs release in `CHANGELOG.md`.

```bash
# Correct — triggers publish
git tag docs/v1.1.0
git push origin docs/v1.1.0

git tag docs/v1.2.0-rc1     # pre-release suffix is fine, still matches docs/v*
git push origin docs/v1.2.0-rc1

# Wrong — these will NOT trigger publish
git tag v1.1.0              # missing docs/ prefix
git tag docs/1.1.0          # missing v
git tag docs-v1.1.0         # wrong separator
```

URL → version mapping after publish:

```
docs.nvidia.com/nemo/curator/latest/...   → symlink to current train (v26.02 today)
docs.nvidia.com/nemo/curator/v26.02/...   → 26.02 train
docs.nvidia.com/nemo/curator/v25.09/...   → 25.09 train
```

## Version Ship Checklist (when cutting a new train)

When the user ships a new version (e.g. `v26.04`):

1. Copy `fern/versions/v26.02/pages/` → `fern/versions/v26.04/pages/` and edit content.
2. Copy `fern/versions/v26.02.yml` → `fern/versions/v26.04.yml` and update all `./v26.02/` path prefixes.
3. Repoint the symlink: `ln -sf v26.04.yml fern/versions/latest.yml`.
4. Update `fern/docs.yml` `versions:` list — add the new display-name, mark older trains stable.
5. Add redirect rules in `fern/docs.yml` for `/nemo/curator/26.04/:path*` → `/nemo/curator/v26.04/:path*` (see existing patterns).
6. Align `display-name` strings with `CHANGELOG.md` and `nemo_curator/package_info.py`.

## Debugging

| Symptom | Fix |
|---|---|
| `fern check` YAML error | 2-space indent; `- page:` inside `contents:`; `path:` is relative to the version YAML file |
| Page 404 in preview | `slug:` missing or duplicated in the same section; confirm in `vXX.YY.yml` |
| `{{ variable }}` shows literally on site | Not in `DEFAULT_VARIABLES` in `substitute_variables.py` — add it there |
| MDX parse error | Replace bare `<https://...>` with `[text](https://...)`; escape `<` in prose with `&lt;` or backticks |
| Old Sphinx URL breaks | Add a `redirects:` entry in `fern/docs.yml` |
| Library reference missing | Run `fern docs md generate` in `fern/` (see `fern/AUTODOCS_GUIDE.md`) |
| Broken image | Path is relative to the MDX file; check `fern/assets/` or `pages/_images/` exists |

## Key References

| File | Purpose |
|---|---|
| `fern/docs.yml` | Site config, versions, redirects, libraries |
| `fern/versions/vXX.YY.yml` | Navigation tree for a version |
| `fern/versions/vXX.YY/pages/` | MDX content for a version |
| `fern/versions/latest.yml` | Symlink → current train's nav (do not edit) |
| `fern/components/` | Custom TSX (footer, release banner) |
| `fern/assets/` | Shared images, SVGs, favicon |
| `fern/substitute_variables.py` | Variable definitions + CI replacement |
| `fern/AUTODOCS_GUIDE.md` | Generating library reference MDX from source |
| `fern/README.md` | Full docs architecture guide |
| `.github/workflows/fern-docs-*.yml` | CI: validation, preview, publish |

---
