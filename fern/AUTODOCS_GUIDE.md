# Fern Library Docs Guide

This guide shows how to add Fern-generated library reference docs to a Fern documentation site.

This guide stays generic so you can reuse it across repositories with minimal path and repository changes.

## When to Use This

Use this setup when you want Fern to:

- pull API docs from a source repository
- generate markdown for a library or package
- include the generated pages in your docs navigation

## Prerequisites

- A Fern docs project with a `fern/` directory
- A source repository that Fern can read
- A package or source directory to document
- Fern CLI installed locally

If needed, upgrade Fern first:

```bash
fern upgrade
```

## 1. Keep `fern.config.json` Minimal

Use `fern.config.json` only for the base Fern configuration.

Example:

```json
{
  "organization": "<your-org>",
  "version": "<fern-cli-version>"
}
```

Do not add `libraries` here. Fern will reject it with an error like:

```text
Unrecognized key(s) in object: 'libraries'
```

## 2. Add the Library Definition to `fern/docs.yml`

Add a `libraries` block to `fern/docs.yml`.

Template:

```yml
libraries:
  <library-name>:
    input:
      git: https://github.com/<org>/<repo>
      subpath: <package-or-source-subpath>
    output:
      path: ./product-docs/<docs-slug>/Library-Reference
    lang: python
```

Example:

```yml
libraries:
  my-library:
    input:
      git: https://github.com/example-org/example-repo
      subpath: src/my_library
    output:
      path: ./product-docs/my-library/Library-Reference
    lang: python
```

Notes:

- `git` points to the repository Fern should read from.
- `subpath` points to the package or source root to document.
- `output.path` is where Fern will write the generated markdown.
- `lang` should match the library language supported by Fern.

## 3. Generate the Library Markdown

Run:

```bash
fern docs md generate
```

Fern will generate markdown under the output path you configured, for example:

```text
fern/product-docs/my-library/Library-Reference
```

Important:

- Run this before previewing the docs.
- If the output folder does not exist yet, navigation entries pointing to it may fail to load.

## 4. Add the Generated Folder to Navigation

Update your versioned nav file, such as `fern/versions/<version>.yml`, and add the generated folder where you want it to appear.

Template:

```yml
- section: API Reference
  contents:
    - folder: ../product-docs/<docs-slug>/Library-Reference
```

Example inside a larger API section:

```yml
- tab: api
  layout:
    - section: API Reference
      slug: reference/api-reference
      contents:
        - page: Overview
          path: ../v1/pages/api-reference/index.mdx
          slug: ""
        - folder: ../product-docs/my-library/Library-Reference
```

The relative path should be correct from the versioned navigation file to the generated output directory.

## 5. Preview the Docs

You can preview locally with either command:

```bash
fern docs dev
```

or

```bash
fern generate --docs --preview
```

If the setup works, your generated library reference pages will appear in the docs navigation.

## 6. Ignore Generated Output

If you do not want to commit generated library docs, add the output directory to `.gitignore`.

Template:

```gitignore
fern/product-docs/<docs-slug>/Library-Reference/
```

Example:

```gitignore
fern/product-docs/my-library/Library-Reference/
```

## Common Issues

### `libraries` in the wrong file

Problem:

```text
Unrecognized key(s) in object: 'libraries'
```

Cause:

- you added `libraries` to `fern.config.json` instead of `fern/docs.yml`

Fix:

- move the `libraries` block into `fern/docs.yml`

### Folder not found during preview

Problem:

```text
Folder not found: ../product-docs/<docs-slug>/Library-Reference
```

Cause:

- the generated markdown does not exist yet
- the relative folder path in navigation is wrong

Fix:

1. run `fern docs md generate`
2. verify the output folder exists
3. verify the relative `folder:` path in the versioned nav file

### Fern version mismatch

Problem:

- features work differently across Fern CLI versions

Fix:

```bash
fern upgrade
```

## Quick Checklist

1. Keep `fern.config.json` minimal.
2. Add `libraries` to `fern/docs.yml`.
3. Run `fern docs md generate`.
4. Add the generated folder to `fern/versions/<version>.yml`.
5. Preview with `fern docs dev` or `fern generate --docs --preview`.
6. Add the generated folder to `.gitignore` if you do not want to check it in.

## Copy-Paste Starter Template

```yml
libraries:
  <library-name>:
    input:
      git: https://github.com/<org>/<repo>
      subpath: <package-or-source-subpath>
    output:
      path: ./product-docs/<docs-slug>/Library-Reference
    lang: python
```

```yml
- section: API Reference
  contents:
    - folder: ../product-docs/<docs-slug>/Library-Reference
```

```gitignore
fern/product-docs/<docs-slug>/Library-Reference/
```
