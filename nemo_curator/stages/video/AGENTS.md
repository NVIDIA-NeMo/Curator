<!-- generated from .stewards/manifest.toml — edit the manifest, not this file -->

# Steward: video

Protect VideoTask metadata and the clip, embedding, preview, and metadata output layout across video pipelines.

Ordinary work: use this map directly with the root map and run only affected checks.
Do not open `.stewards/PROTOCOL.md` or `.stewards/manifest.toml` unless the task is an explicit review/audit or steward-network maintenance.

## Protects

| Invariant | Sev | Backing | Proof / anchor |
| --- | --- | --- | --- |
| VideoTask serialization, metadata, and helper behavior remain covered by the focused task suite. | P1 | machine-backed | `uv run pytest tests/tasks/test_video.py -q -m 'not gpu'` (`video-task`) |
| ClipWriterStage owns explicit paths for processed videos, clip chunks, clips, previews, metadata, and embedding artifacts. | P1 | manual | nemo_curator/stages/video/io/clip_writer.py · `class ClipWriterStage` |
| GPU acceleration claims distinguish measured supported paths from roadmap goals and CPU fallbacks. | P2 | none | — |

## Guardrails

- Describe current CPU/GPU paths and writer formats precisely; do not turn acceleration goals into present-tense guarantees.

## Edges

- depends-on → **pipeline** (stage and task contracts)
- depends-on → **backends** (distributed and accelerator execution)

## Owns

- **code:** `nemo_curator/stages/video`, `nemo_curator/tasks/video.py`
- **tests:** `tests/stages/video`, `tests/tasks/test_video.py`
