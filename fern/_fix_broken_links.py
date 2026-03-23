#!/usr/bin/env python3
"""One-off: rewrite internal links in Fern MDX to match versions/*.yml routes. Not imported."""

from __future__ import annotations

import re
from pathlib import Path

REPO_FERN = Path(__file__).resolve().parent

# Longest-first replacement (path fragments and full hrefs)
REPLACEMENTS: list[tuple[str, str]] = [
    # API docs tab paths (slug reference/api-reference)
    ("/api/reference/api-reference/", "/reference/api-reference/"),
    # Reference infrastructure → infra nested slugs
    ("/reference/infrastructure/memory-management", "/reference/infra/memory/management"),
    ("/reference/infrastructure/gpu-processing", "/reference/infra/gpu/processing"),
    ("/reference/infrastructure/resumable-processing", "/reference/infra/resumable/processing"),
    ("/reference/infrastructure/container-environments", "/reference/infra/container/environments"),
    ("/reference/infrastructure/execution-backends", "/reference/infra/execution-backends"),
    # Legacy /docs/* → Fern paths
    ("/docs/reference/infrastructure/container/environments", "/reference/infra/container/environments"),
    ("/docs/reference/execution/backends", "/reference/infra/execution-backends"),
    ("/docs/reference/execution/backends", "/reference/infra/execution-backends"),
    ("/docs/admin/deployment/requirements", "/admin/deployment/requirements"),
    ("/docs/get-started/text", "/get-started/text"),
    ("/docs/get-started/image", "/get-started/image"),
    ("/docs/get-started/audio", "/get-started/audio"),
    ("/docs/get-started/video", "/get-started/video"),
    ("/docs/migration/guide", "/about/release-notes/migration-guide"),
    ("/docs/migration/faq", "/about/release-notes/migration-faq"),
    ("/docs/about/concepts/text/data/acquisition", "/about/concepts/text/data/acquisition"),
    ("/docs/about/concepts/text/data/loading", "/about/concepts/text/data/loading"),
    ("/docs/about/concepts/text/data/processing", "/about/concepts/text/data/processing"),
    ("/docs/about/concepts/video/architecture", "/about/concepts/video/architecture"),
    ("/docs/about/concepts/video/abstractions", "/about/concepts/video/abstractions"),
    ("/docs/about/concepts/audio/asr/pipeline", "/about/concepts/audio/asr/pipeline"),
    ("/docs/about/concepts/deduplication", "/about/concepts/deduplication"),
    ("/docs/text/process/data/filter/heuristic", "/curate-text/process-data/quality-assessment/heuristic"),
    ("/docs/text/process/data/dedup", "/curate-text/process-data/deduplication"),
    # About concepts: flat filenames → nested section slugs (v26 nav)
    ("/about/concepts/text/data-loading-concepts", "/about/concepts/text/data/loading"),
    ("/about/concepts/text/data-processing-concepts", "/about/concepts/text/data/processing"),
    ("/about/concepts/text/data-acquisition-concepts", "/about/concepts/text/data/acquisition"),
    ("/about/concepts/text/data-curation-pipeline", "/about/concepts/text/curation/pipeline"),
    ("/about/concepts/image/data-loading-concepts", "/about/concepts/image/data/loading"),
    ("/about/concepts/image/data-processing-concepts", "/about/concepts/image/data/processing"),
    ("/about/concepts/image/data-export-concepts", "/about/concepts/image/data/export"),
    ("/about/concepts/video/data-flow", "/about/concepts/video/data/flow"),
    ("/about/concepts/audio/manifests-ingest", "/about/concepts/audio/manifests/ingest"),
    ("/about/concepts/audio/asr-pipeline", "/about/concepts/audio/asr/pipeline"),
    ("/about/concepts/audio/quality-metrics", "/about/concepts/audio/quality/metrics"),
    ("/about/concepts/audio/curation-pipeline", "/about/concepts/audio/curation/pipeline"),
    ("/about/concepts/audio/audio-batch", "/about/concepts/audio/audio/batch"),
    ("/about/concepts/audio/text-integration", "/about/concepts/audio/text/integration"),
    ("/about/concepts/image-data-processing", "/about/concepts/image/data/processing"),
    # Short / legacy slugs → full routes (v26)
    ("/reference-infrastructure-container-environments", "/reference/infra/container/environments"),
    ("/admin-deployment-requirements", "/admin/deployment/requirements"),
    ("/deployment/requirements", "/admin/deployment/requirements"),
    ("/migration-guide", "/about/release-notes/migration-guide"),
    ("/migration-faq", "/about/release-notes/migration-faq"),
    ("/gs-text", "/get-started/text"),
    ("/gs-image", "/get-started/image"),
    ("/gs-audio", "/get-started/audio"),
    ("/gs-video", "/get-started/video"),
    ("/admin-overview", "/admin"),
    ("/text-process-data-filter-heuristic", "/curate-text/process-data/quality-assessment/heuristic"),
    ("/text-process-data-filter", "/curate-text/process-data/quality-assessment/heuristic"),
    ("/text-process-data-dedup-exact", "/curate-text/process-data/deduplication/exact"),
    ("/text-process-data-dedup-fuzzy", "/curate-text/process-data/deduplication/fuzzy"),
    ("/text-process-data-format-sem-dedup", "/curate-text/process-data/deduplication/semdedup"),
    ("/text-process-data-dedup", "/curate-text/process-data/deduplication"),
    ("/text-tutorials", "/curate-text/tutorials"),
    ("/text-load-data-read-existing", "/curate-text/load-data/read-existing"),
    ("/text-load-data", "/curate-text/load-data"),
    ("/text-overview", "/curate-text"),
    ("/image-overview", "/curate-images"),
    ("/image-tutorials-dedup", "/curate-images/tutorials/dedup-workflow"),
    ("/image-tutorials-beginner", "/curate-images/tutorials/beginner"),
    ("/image-process-data-embeddings-clip", "/curate-images/process-data/embeddings/clip-embedder"),
    ("/image-process-data-filters-aesthetic", "/curate-images/process-data/filters/aesthetic"),
    ("/video-overview", "/curate-video"),
    ("/video-save-export", "/curate-video/save-export"),
    ("/video-process-clipping", "/curate-video/process-data/clipping"),
    ("/video-process-transcoding", "/curate-video/process-data/transcoding"),
    ("/video-process-frame-extraction", "/curate-video/process-data/frame-extraction"),
    ("/video-process-embeddings", "/curate-video/process-data/embeddings"),
    ("/video-process-filtering", "/curate-video/process-data/filtering"),
    ("/video-process-filtering-aesthetic", "/curate-images/process-data/filters/aesthetic"),
    ("/video-process-captions-preview", "/curate-video/process-data/captions-preview"),
    ("/video-process-dedup", "/curate-video/process-data/dedup"),
    ("/video-tutorials-split-dedup", "/curate-video/tutorials/split-dedup"),
    ("/video-tutorials-pipeline-cust-add-code", "/curate-video/tutorials/pipeline-customization/add-cust-code"),
    ("/video-tutorials-pipeline-cust-env", "/curate-video/tutorials/pipeline-customization/add-cust-env"),
    ("/video-tutorials-pipeline-cust-add-stage", "/curate-video/tutorials/pipeline-customization/add-cust-stage"),
    ("/video-tutorials-pipeline-cust-add-model", "/curate-video/tutorials/pipeline-customization/add-cust-model"),
    ("/duration-filtering", "/curate-audio/process-data/quality-assessment/duration-filtering"),
    ("/quality-assessment/duration-filtering", "/curate-audio/process-data/quality-assessment/duration-filtering"),
    ("/format-validation", "/curate-audio/process-data/audio-analysis/format-validation"),
    # Relative .md / Sphinx paths → Fern routes
    ("../../../reference/infrastructure/resumable-processing.md", "/reference/infra/resumable/processing"),
    ("../../reference/infrastructure/gpu-processing.md", "/reference/infra/gpu/processing"),
    ("../../reference/infrastructure/execution-backends.md", "/reference/infra/execution-backends"),
    ("../../curate-video/index.md", "/curate-video"),
    ("../../curate-video/process-data/clipping.md", "/curate-video/process-data/clipping"),
    ("../../curate-video/process-data/dedup.md", "/curate-video/process-data/dedup"),
    ("../../curate-video/process-data/filtering.md", "/curate-video/process-data/filtering"),
    ("../../curate-video/process-data/captions-preview.md", "/curate-video/process-data/captions-preview"),
    ("../../curate-audio/index.md", "/curate-audio"),
    ("../../curate-audio/process-data/asr-inference/index.md", "/curate-audio/process-data/asr-inference"),
    ("../../curate-audio/process-data/quality-assessment/index.md", "/curate-audio/process-data/quality-assessment"),
    ("../../curate-audio/process-data/audio-analysis/index.md", "/curate-audio/process-data/audio-analysis"),
    ("../../curate-audio/process-data/text-integration/index.md", "/curate-audio/process-data/text-integration"),
    ("../../curate-text/index.md", "/curate-text"),
    ("../../curate-text/process-data/quality-assessment/distributed-classifier.md", "/curate-text/process-data/quality-assessment/distributed-classifier"),
    ("../../curate-text/process-data/deduplication/semdedup.md", "/curate-text/process-data/deduplication/semdedup"),
    ("../../curate-images/index.md", "/curate-images"),
    ("../../curate-images/process-data/embeddings/index.md", "/curate-images/process-data/embeddings"),
    ("../../curate-images/process-data/filters/aesthetic.md", "/curate-images/process-data/filters/aesthetic"),
    ("../../curate-images/process-data/filters/nsfw.md", "/curate-images/process-data/filters/nsfw"),
    ("../concepts/video/architecture.md", "/about/concepts/video/architecture"),
    ("../../apidocs/index.rst", "https://docs.nvidia.com/nemo/curator/latest/py-modindex.html"),
]

REPLACEMENTS.sort(key=lambda x: -len(x[0]))

APIDOC_BASE = "https://docs.nvidia.com/nemo/curator/latest/apidocs"


def fix_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    orig = text

    for old, new in REPLACEMENTS:
        text = text.replace(old, new)

    # /../apidocs/... → NVIDIA hosted Sphinx
    def apidoc_repl(m: re.Match[str]) -> str:
        rest = m.group(1).lstrip("/")
        return f'"{APIDOC_BASE}/{rest}"'

    text = re.sub(r'"/\.\./apidocs/([^"]+)"', apidoc_repl, text)
    text = re.sub(r"'/\.\./apidocs/([^']+)'", lambda m: f"'{APIDOC_BASE}/{m.group(1).lstrip('/')}'", text)

    # Leading-slashless curate-* paths in href or ](
    text = re.sub(
        r'(\]\(|href=")(curate-[a-z-]+/[^")\s]+)',
        lambda m: m.group(1) + "/" + m.group(2),
        text,
    )
    # about/ without leading slash (not already fixed)
    text = re.sub(
        r'(\]\(|href=")(about/concepts/[^")\s]+)',
        lambda m: m.group(1) + "/" + m.group(2),
        text,
    )

    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    roots = [
        REPO_FERN / "versions/v25.09/pages",
        REPO_FERN / "versions/v26.02/pages",
    ]
    n = 0
    for root in roots:
        for mdx in sorted(root.rglob("*.mdx")):
            if fix_file(mdx):
                n += 1
                print(mdx.relative_to(REPO_FERN))
    print(f"Updated {n} files")


if __name__ == "__main__":
    main()
