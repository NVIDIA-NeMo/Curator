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
│   ├── latest.yml            # Symlink → v26.04.yml (do not edit directly)
│   ├── v26.04.yml            # Nav tree for current train
│   ├── v26.04/pages/         # MDX content for current train
│   ├── v25.09.yml
│   └── v25.09/pages/
├── components/               # Custom TSX components (footer, etc.)
├── assets/                   # Images, SVGs, favicon
├── substitute_variables.py   # CI: resolves {{ variables }} in MDX
└── AUTODOCS_GUIDE.md         # Library reference generation guide
```

**Current train:** `v26.04`. Default all new pages there unless the user specifies a version.

```
File system                              Published URL
───────────────────────────────────────  ────────────────────────────────────────
fern/versions/v26.04/pages/              docs.nvidia.com/nemo/curator/latest/
  └─ get-started/text.mdx                  └─ get-started/text
fern/versions/v26.04.yml ── nav for ──┐  docs.nvidia.com/nemo/curator/v26.04/
fern/versions/latest.yml ─ symlink ───┘    └─ get-started/text
fern/versions/v25.09/pages/              docs.nvidia.com/nemo/curator/v25.09/
  └─ get-started/text.mdx                  └─ get-started/text
```

## Operations

### Add a Page

1. Gather: page title, target section, filename (kebab-case `.mdx`), subdirectory under `fern/versions/v26.04/pages/`.
2. Create `fern/versions/v26.04/pages/<subdirectory>/<filename>.mdx`:

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

3. Add a nav entry in `fern/versions/v26.04.yml` under the correct section:

```yaml
- page: <Page Title>
  path: ./v26.04/pages/<subdirectory>/<filename>.mdx
  slug: <filename>
```

4. If this also applies to `latest`, no action needed — `latest.yml` is a symlink to `v26.04.yml`.

### Update a Page

1. Locate by path, title, or keyword (`grep -rn` in `fern/versions/v26.04/pages/`).
2. **Content only** — edit the MDX directly.
3. **Title change** — update the frontmatter and the `- page:` name in `fern/versions/v26.04.yml`.
4. **Section move** — `git mv` the file, update its `path:` in the nav, and fix all incoming links.
5. **Slug change** — update `slug:` in the nav and add a redirect in `fern/docs.yml` so old URLs keep working.

### Remove a Page

1. Find incoming links: `grep -r "<filename>" fern/versions/v26.04/pages/ --include="*.mdx"`.
2. `git rm fern/versions/v26.04/pages/<subdirectory>/<filename>.mdx`.
3. Remove the `- page:` block from `fern/versions/v26.04.yml`. If it was the last page in a section, remove the `- section:` block.
4. Fix or remove all incoming links found in step 1.
5. Add a redirect in `fern/docs.yml` if the URL was public.

### Back-port to an Older Version

Only when explicitly asked. Repeat the operation in the corresponding `fern/versions/vXX.YY/` tree and `vXX.YY.yml` nav. MDX content often diverges between trains — do not blindly copy.

### Worked Example: Adding a Page

Request: *"Add a how-to for benchmarking text pipelines under Curate Text."*

1. Create `fern/versions/v26.04/pages/curate-text/benchmarking.mdx`:

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

2. Add nav entry in `fern/versions/v26.04.yml` under the existing `Curate Text` section:

   ```yaml
   - page: Benchmark Text Pipelines
     path: ./v26.04/pages/curate-text/benchmarking.mdx
     slug: benchmarking
   ```

3. `cd fern && fern check` then `fern docs dev` and verify the page renders at `/curate-text/benchmarking`.

### Worked Example: Renaming a Slug (with Redirect)

Request: *"Rename `/curate-text/benchmarking` to `/curate-text/performance`."*

1. Update `slug:` in `fern/versions/v26.04.yml`: `slug: performance`.
2. (Optional) `git mv` the MDX file if you want the filename to match the slug.
3. Add a redirect to `fern/docs.yml` so old links keep working:

   ```yaml
   redirects:
     - source: "/nemo/curator/latest/curate-text/benchmarking"
       destination: "/nemo/curator/latest/curate-text/performance"
     - source: "/nemo/curator/v26.04/curate-text/benchmarking"
       destination: "/nemo/curator/v26.04/curate-text/performance"
   ```

4. `grep -rn "/curate-text/benchmarking" fern/versions/v26.04/pages/` and update any incoming links.

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
python fern/substitute_variables.py versions/v26.04 --version 26.04 --dry-run
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

**DCO sign-off (`-s`) is required** on every commit. CI enforces it. If you forget, amend with `git commit --amend --no-edit -s` and force-push the branch.

PRs that touch `fern/**` get an automatic Fern preview URL posted as a comment by `.github/workflows/fern-docs-preview-comment.yml`. No manual step needed.

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
docs.nvidia.com/nemo/curator/latest/...   → symlink to current train (v26.04 today)
docs.nvidia.com/nemo/curator/v26.04/...   → 26.04 train
docs.nvidia.com/nemo/curator/v26.02/...   → 26.02 train
docs.nvidia.com/nemo/curator/v25.09/...   → 25.09 train
```

## Version Ship Checklist (when cutting a new train)

When the user ships a new version (e.g. cutting `v26.06` while `v26.04` is current):

1. Copy `fern/versions/v26.04/pages/` → `fern/versions/v26.06/pages/` and edit content.
2. Copy `fern/versions/v26.04.yml` → `fern/versions/v26.06.yml` and update all `./v26.04/` path prefixes to `./v26.06/`.
3. Repoint the symlink: `ln -sf v26.06.yml fern/versions/latest.yml`.
4. Update `fern/docs.yml` `versions:` list — add the new display-name, mark older trains stable.
5. Add redirect rules in `fern/docs.yml` for `/nemo/curator/26.06/:path*` → `/nemo/curator/v26.06/:path*` (see existing patterns).
6. Add `*/index.html` redirect for the new version (e.g. `/nemo/curator/v26.06/index.html` → `/nemo/curator/v26.06`). The `:path*` rule does **not** match the empty-path case, so each version-root index.html needs its own explicit rule.
7. Align `display-name` strings with `CHANGELOG.md` and `nemo_curator/package_info.py`.

## Holding a Version Back from Publish

A version is included in the published site only when it appears in the `versions:` block of `fern/docs.yml`. The MDX tree (`fern/versions/vXX.YY/`) and nav file (`fern/versions/vXX.YY.yml`) can sit in the repo unpublished — Fern doesn't auto-publish every YAML it finds.

**To stage a version without publishing it** (e.g. work-in-progress on `v26.06` while `v26.04` is current):

```yaml
# fern/docs.yml
versions:
  - display-name: "Latest · v1.1.2 (26.04)"
    path: versions/latest.yml
    slug: latest
  - display-name: "26.04 · v1.1.2"
    path: versions/v26.04.yml
    slug: v26.04
  # v26.06 staged in repo but not listed here → not published
```

**To pull an already-shipping version back** (e.g. hold `v26.04` while pushing fixes to older trains):

1. Remove the `v26.04` entry from `versions:` in `fern/docs.yml`.
2. If `latest` should also stop serving 26.04 content, repoint the symlink: `ln -sf v26.02.yml fern/versions/latest.yml`. Otherwise leave `latest` alone — it will keep serving v26.04 content under `/latest/` even with `v26.04` removed (since `latest.yml` references the v26.04 pages directly).
3. Tag and push `docs/v*` to publish.
4. Restore the entry (and symlink) when ready.

This is a temporary maneuver — track the change so it gets reverted.

**Audiences (alternative):** Fern supports `audiences:` on versions plus separate `instances:` (e.g. staging vs production). This is heavier setup — only adopt if multi-instance publishing is genuinely needed. NeMo Curator does not currently configure instances. References: [Fern versioning](https://buildwithfern.com/learn/docs/configuration/versioning), [Fern publishing](https://buildwithfern.com/learn/docs/configuration/publishing).

**Do not use `hidden: true` to hide a version from publish.** Hidden versions are removed from navigation/search/indexing but remain accessible by direct URL — still effectively published.

## Library Reference (Autodocs) and the Fern Cross-Ref Bug

`fern/docs.yml` declares a `libraries:` block that pulls Python source from `nemo_curator/` and generates MDX into `fern/product-docs/nemo-curator/Full-Library-Reference/` (gitignored). It runs as `fern docs md generate` in the publish and preview workflows.

**Known bug in the Fern Python library generator** (filed upstream): the generator emits cross-references that miss the `/nemo/curator` site basepath (links use `/nemo-curator/...` instead of `/nemo/curator/nemo-curator/...`) and tacks on Sphinx-style `#nemo_curator-…` fragments that don't match any rendered anchor. Result: ~540 broken links across the generated API reference.

**No in-repo workaround currently.** A post-generation rewrite (walking the generated MDX, fixing the basepath, dropping stale fragments) is feasible but not yet wired up. Track the upstream Fern fix; revisit if it doesn't land soon.

`fern/_fix_broken_links.py` separately rewrites a long list of legacy URL patterns (`/api/reference/api-reference/`, old Sphinx slugs, etc.) on the **committed** MDX under `fern/versions/v25.09/pages/` and `fern/versions/v26.02/pages/`. CI does not run it, so committed pages can drift. Re-run locally and commit the diff if you see drift:

```bash
python3 fern/_fix_broken_links.py
```

## Redirect Quirks

- **`:path*` does not match the empty-path case.** `/nemo/curator/:path*/index.html` will not catch `/nemo/curator/index.html` — that needs its own explicit rule. Same for every version-root: `/nemo/curator/{latest,vXX.YY}/index.html` each need a dedicated entry. Pattern: define the explicit empty-path rules **before** the `:path*` rule.
- **Order matters.** Fern processes redirects top-down, first match wins. Put more specific rules above catch-alls.
- **Version slugs** in `fern/docs.yml` use the `vXX.YY` form (e.g. `v26.04`). Calendar-train forms (`26.04`) need redirects to the `v`-prefixed slug.

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
