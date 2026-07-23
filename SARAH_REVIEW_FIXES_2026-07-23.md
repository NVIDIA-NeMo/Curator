# Sarah Review Fixes — 2026-07-23

## Scope

This document covers every PR #1967 review thread authored by
`sarahyurick` in the seven-day window beginning 2026-07-16 UTC. The review
data was read with thread state and inline locations so unresolved, resolved,
current, and outdated feedback could be distinguished.

## Review disposition

| Review feedback | Disposition |
| --- | --- |
| Consider `NoneTask` for the empty `ASRStage.process_batch` branch and verify audio resumability. | Removed the redundant empty-batch branch so the normal 0-to-0 batch path applies. A zero-input call has no parent slot to represent with `NoneTask`; creating one would produce an unparented sentinel and no resumability delta. Added a backend-adapter regression test proving that an unsupported-language audio input remains a real positional task, inherits its source lineage, emits the expected zero delta at this non-sink stage, and is not sent to the model adapter. |
| Move `_process_plain_batch` into `process_batch`. | Already addressed by commit `16f0c441`: the helper was removed and its logic was inlined. |
| Use a top-level `librosa` import in the waveform helper. | Moved `librosa` to module scope. The ASR model package still lazily imports the concrete Qwen adapter, so importing the model-neutral ASR protocol does not load this stack. |
| Remove `tests/config/test_qwen_reference_parity.py`. | Already addressed by commit `16f0c441`: the file is absent from the current PR. |
| Do not add a copyright header to an empty ASR package file. | The reviewed empty package file was superseded by commit `16f0c441`. `nemo_curator/models/asr/__init__.py` now contains the package's lazy public exports, so it is no longer empty and its license header remains appropriate. |
| Remove generated section-banner comments from the ASR stage tests. | Already addressed by commit `a3d2bedd`: the reviewed banners were removed. |
| Remove the blank line from `tests/models/asr/__init__.py`. | Made the tracked file zero bytes. |
| Use a broader vLLM-specific optional-extra name. | Renamed `audio_qwen` to `audio_vllm` in `pyproject.toml`, `uv.lock`, runtime installation guidance, error messages, and tutorial documentation. |
| Reconsider the adapter-local `ThreadPoolExecutor` in favor of Curator resource/worker allocation. | Removed `ThreadPoolExecutor`, the `prep_workers` option, pool lifecycle state, and parallel map calls. Turn-1 and Turn-2 input preparation now run deterministically inside the Curator-managed adapter worker, with strict length checks on every paired input list. |

## Verification

- Ruff passed on all edited Python files.
- Qwen adapter, waveform, lazy-import, ASR stage, and resumability tests:
  `77 passed`.
- `uv lock --check` passed.
- `git diff --check` passed.

No live Kratos comparison was launched: these changes are intentionally local
and the requested workflow forbids pushing the commit needed by the image
builder.
