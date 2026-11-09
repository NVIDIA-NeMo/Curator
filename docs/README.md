# NeMo Curator docs

The Sphinx tree that used to live here has been retired. NeMo Curator's documentation is now authored in [Fern](https://buildwithfern.com/) MDX under [`../fern/`](../fern/) and published to **[docs.nvidia.com/nemo/curator](https://docs.nvidia.com/nemo/curator)**.

- **Edit pages:** see [`../fern/README.md`](../fern/README.md) for layout, local dev, and conventions.
- **Add a page:** drop an MDX file under `fern/versions/main/pages/` (the bleeding-edge tree, published at `/main/...` with `availability: beta`) and wire it into `fern/versions/main.yml`. Back-port to `fern/versions/v26.04/` only when the fix needs to ship to the current GA.
- **Release notes:** add to `fern/versions/main/pages/about/release-notes/` — never to this directory.
- **Preview a PR:** PRs touching `fern/**` get an automatic 🌿 preview URL posted as a comment.

Old `/nemo/curator/...` URLs (including legacy `index.html` and `.html` paths from the Sphinx build) are redirected to their Fern equivalents via `redirects:` in `fern/docs.yml`. If you find a broken link to the published site, add a redirect there.
