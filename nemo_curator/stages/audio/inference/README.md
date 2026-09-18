# Local Duration Bucketing for Audio GPU Inference

This document describes the local duration-batching contract implemented by
[`ASRStage`](asr/stage.py) and how to apply the same pattern to another audio
GPU inference stage.

The contract has three controls:

| Field | Default | Purpose |
|---|---:|---|
| `max_audio_sec_per_actor` | required | Maximum padded audio seconds planned for one adapter call |
| `max_inference_duration_s` | `2400` | Model-specific maximum duration of one segment |
| `local_bucketing` | `false` | Stable-sort by duration and optimize call boundaries inside the current planning window |

There are no bucket edges, per-bucket limits, item-count caps, timers, queues,
flush operations, generic cost features, or separate adapter batch-size
setting. `batch_size` remains a backend candidate-window setting; it does not
cap the number of segments in an adapter call.

## Mental model

Audio tensors in one GPU call are commonly padded to the longest item. For a
nonempty call with durations `d[0] ... d[n-1]`, define:

```text
useful audio seconds = sum(d)
padded audio seconds = n * max(d)
padding efficiency   = sum(d) / (n * max(d))
```

`ASRStage` uses `padded audio seconds` as a simple capacity proxy. Every
planned adapter call must satisfy:

```text
number of items in the call * longest item duration
    <= max_audio_sec_per_actor
```

This proxy captures the main cost of padding more directly than summing raw
durations. It is still not a proof of GPU-memory safety: model architecture,
precision, decoder state, framework workspaces, and fixed per-item overhead
also matter. The value must be measured and tuned for each adapter, model, and
hardware configuration.

Despite its name, `max_audio_sec_per_actor` limits each adapter invocation
planned by an actor. It is not a lifetime quota and does not accumulate across
`process_batch()` calls.

## Scope and data flow

The complete planning horizon is exactly one finite `process_batch(tasks)`
call:

```text
backend candidate rows (`batch_size`)
    -> eligibility checks and waveform preparation
    -> model-safe segmentation of every eligible parent
    -> one flat list of segments from all eligible parents
    -> optional stable duration ordering
    -> padded-seconds packing
    -> adapter calls
    -> scatter segment results to original positions
    -> stitch segments for each parent
    -> return parent rows in original order
```

The distinctions in that flow are important:

1. A **parent row** is one `AudioTask` received from the backend.
2. A **segment** is one model-safe waveform derived from a parent. One parent
   may create several segments.
3. An **adapter call** is one capacity-bounded group of segments. It may
   contain segments from several parents, including a short remainder from a
   long parent.

Skipped languages, reused outputs, and waveforms that fail preparation do not
enter the segment plan. All other segments produced from all input rows in the
same `process_batch()` call share one local planning window. Nothing is held
for a later call, another actor, or another worker.

## Segmentation always happens first

`local_bucketing` controls reordering only. It never controls segmentation.

After decode, downmix, and resampling, `ASRStage` always calls
[`plan_audio_segments`](../model_input_segmentation.py) with
`max_inference_duration_s`. For sample rate `s` and model limit `D`, the
maximum samples in one segment are:

```text
max_samples = max(1, int(D * s))
```

The intervals are contiguous, nonoverlapping, and cover the full waveform.
An exact multiple creates no empty tail, and every nonempty remainder becomes
another segment. A zero-sample waveform remains representable as one
zero-duration segment.

The stage derives `audio_seconds` from the actual prepared segment, not from a
possibly stale manifest duration:

```python
audio_seconds = number_of_segment_samples / target_sample_rate
```

`max_inference_duration_s` must be less than or equal to
`max_audio_sec_per_actor`. That construction-time validation guarantees that
one model-safe segment can fit in one planned call. Choose the model limit for
model correctness first, then choose an actor budget large enough to contain
it.

## Packing algorithm

Planning is deterministic, but the two modes use different boundary planners.
Both enforce the padded-audio budget on every adapter call. Durations need no
second validation here: they are produced internally from model-safe segments.

With `local_bucketing=false`, the planner preserves input order and scans it
greedily. It extends the current call while the candidate padded cost fits,
emits that call when the next segment would exceed the budget, and then starts
the next call with that segment.

With `local_bucketing=true`, the planner:

1. Stable-sorts all segments by ascending duration. Equal-duration segments
   retain their original relative order.
2. Treats every adapter call as one contiguous span of that sorted sequence.
   For a span `[start, stop)`, its padded cost is
   `(stop - start) * duration[stop - 1]`.
3. Uses dynamic programming from the end of the sequence to score every
   feasible next boundary.
4. Minimizes the score lexicographically: first the total number of adapter
   calls, then the sum of padded seconds across those calls.
5. Reconstructs the selected spans, executes them in sorted-plan order, and
   scatters each result through its saved pre-planning segment index.

The enabled planner is therefore the exact optimum for its stable
duration-sorted, contiguous-span planning space. It is not a static bucket
scheme or a scheduler across process calls, actors, or workers. With bucketing
off, the same budget still applies; only reordering and dynamic-programming
boundary optimization are disabled.

### Worked example

Suppose the segment durations in original order are:

```text
indices:    [0, 1, 2, 3]
durations:  [8, 2, 7, 3]
budget:     16 padded audio seconds
```

Without local bucketing, original-order packing produces:

```text
[8, 2] -> 2 * 8 = 16 padded seconds
[7, 3] -> 2 * 7 = 14 padded seconds
```

With local bucketing, stable ascending order is indices `[1, 3, 2, 0]`:

```text
[2, 3] -> 2 * 3 = 6 padded seconds
[7, 8] -> 2 * 8 = 16 padded seconds
```

The same 20 useful audio seconds require 30 proxy seconds without reordering
and 22 with reordering. The adapter results are then scattered to indices
`[0, 1, 2, 3]`, so model-call order cannot reorder output rows.

### Why the enabled planner uses dynamic programming

Sorted greedy packing does not always minimize padding when several plans use
the same number of calls. For sorted durations `[1, 2, 2]` and a budget of `4`,
a greedy scan would produce:

```text
[1, 2] -> 2 * 2 = 4 padded seconds
[2]    -> 1 * 2 = 2 padded seconds
score  -> 2 calls, 6 padded seconds
```

The dynamic-programming planner instead selects:

```text
[1]    -> 1 * 1 = 1 padded second
[2, 2] -> 2 * 2 = 4 padded seconds
score  -> 2 calls, 5 padded seconds
```

Three singleton calls would also total 5 padded seconds, but they lose on the
primary call-count objective. Thus the selected `[1]`, `[2, 2]` plan is the
lexicographic optimum. Python tuple ordering implements the calls-first,
padded-seconds-second comparison directly. Exact score ties keep the first
boundary encountered. Budget feasibility alone uses a small floating-point
tolerance (`rel_tol=1e-12`, `abs_tol=1e-9`), so decimal boundaries such as
three 0.1-second items under a 0.3-second budget do not split spuriously.

## Configuration

### Python

```python
from nemo_curator.stages.audio.inference.asr.stage import ASRStage

asr = ASRStage(
    adapter_target="nemo_curator.models.asr.nemo_asr.NeMoASRAdapter",
    model_id="nvidia/stt_en_fastconformer_ctc_large",
    audio_filepath_key="resampled_audio_filepath",
    max_audio_sec_per_actor=240,
    max_inference_duration_s=120,
    local_bucketing=True,
    batch_size=32,
)
```

### Hydra YAML

```yaml
- _target_: nemo_curator.stages.audio.inference.asr.stage.ASRStage
  adapter_target: nemo_curator.models.asr.nemo_asr.NeMoASRAdapter
  model_id: nvidia/stt_en_fastconformer_ctc_large
  audio_filepath_key: resampled_audio_filepath
  max_audio_sec_per_actor: 240
  max_inference_duration_s: 120
  local_bucketing: true
  batch_size: 32
```

Set `local_bucketing: false` for the original-order control. Do not remove
`max_audio_sec_per_actor`: it is required and bounds calls in both modes.

The example budget permits either two 120-second segments, four 60-second
segments, or twenty-four 10-second segments when those are the longest items
in their respective calls. Actual group membership depends on all durations
in the current candidate window.

## What each control does

| Control | Boundary | Effect |
|---|---|---|
| `batch_size` | Backend to stage | Maximum candidate parent rows normally supplied to one `process_batch()` call; a larger window offers more regrouping choices but uses more host memory |
| `max_inference_duration_s` | Parent waveform to segment | Always splits prepared audio at the model-specific single-input limit |
| `local_bucketing` | Segment planning order | `true` uses stable ascending duration order; `false` keeps original order |
| `max_audio_sec_per_actor` | Planned adapter call | Caps the padded-seconds proxy in both ordering modes |
| Adapter-native settings | Inside adapter/model | Preserve any model-specific decoding or implementation controls |

In particular, `batch_size` is not reused as an adapter item cap. A backend
window of three parent rows may yield one adapter call, several adapter calls,
or more than three segments after long-audio splitting.

## Tuning

There is no universal budget. Tune with the exact model, precision, GPU,
decoder settings, and post-segmentation duration distribution.

1. Establish the largest semantically safe single input and set
   `max_inference_duration_s`. Segmentation and stitching behavior is part of
   the model contract, not a memory-tuning trick.
2. Start `max_audio_sec_per_actor` from a known-safe uniform call. If `k`
   clips of duration `d` are safe, `k * d` is a reasonable initial proxy
   budget. Keep it at least as large as `max_inference_duration_s`.
3. Exercise the longest allowed segment by itself. Then increase the budget
   gradually while measuring peak GPU memory, throughput, and failures.
4. Use representative short, medium, long, exact-boundary, and remainder
   inputs. Average duration alone hides padding and tail behavior.
5. Compare `local_bucketing=false` and `true` with the same candidate windows,
   inputs, model settings, and hardware. Validate outputs before comparing
   speed.
6. Tune `batch_size` separately. Too small gives the local optimizer few
   alternatives; too large increases decode/preparation latency and host
   memory because all candidate waveforms are prepared before model execution.

A duration budget cannot express fixed per-item memory. Very short or
zero-duration items may therefore produce a large call from a finite candidate
window. If an adapter has a hard native count limit, it must enforce or expose
that model-specific constraint at its actual model boundary; do not silently
reinterpret `batch_size` as that limit.

## Correctness invariants

An implementation is correct only when all of these properties hold:

1. Model-safe segmentation runs whether local bucketing is on or off.
2. Every eligible segment has one duration computed after final waveform
   preparation.
3. Every eligible segment is submitted exactly once.
4. Every nonempty adapter call satisfies
   `len(call) * max(duration) <= max_audio_sec_per_actor`.
5. Bucketing off greedily packs the original segment order. Bucketing on uses
   stable ascending duration order and chooses the contiguous partition with
   the fewest calls and then the fewest total padded seconds.
6. Each adapter call returns exactly one result per submitted segment.
7. Results are scattered to original segment positions before parent
   assembly.
8. Each parent's segment results are stitched in temporal order.
9. Parent output order and skip/error behavior do not depend on execution
   order.
10. No pending audio, planner state, timer, or flush obligation survives the
    current `process_batch()` call.

The persistent worker-local adapter and model are unrelated to planner state;
they may live across calls as usual.

## Applying the pattern to another audio GPU stage

The planner is currently implemented inside `ASRStage`; it is not a generic
base-class hook. Another stage can adopt the approach when its model exposes a
real batched call across independent audio inputs and its results can be
mapped back unambiguously.

Use this integration sequence:

1. Add the same required padded-seconds budget and local-bucketing Boolean to
   the stage. Keep `batch_size` as its backend window.
2. In one `process_batch()`, validate and prepare all eligible parents before
   planning model calls.
3. Apply any model-specific segmentation unconditionally. Do not copy ASR's
   split-and-text-stitch semantics unless they are valid for that model.
4. Flatten prepared model inputs and save, for each item, its original item
   index, parent index, segment ordinal, and any timestamp or side-effect
   metadata needed later.
5. Compute duration from the final resampled or sliced waveform.
6. Preserve input order when local bucketing is off; stable-sort by ascending
   duration when it is on.
7. With bucketing off, greedily pack the input-order sequence. With bucketing
   on, use dynamic programming over sorted contiguous spans to minimize call
   count and then total padded seconds, using
   `span_count * span_max_duration` as each span's cost.
8. Call the model, require a complete result mapping, scatter by saved item
   index, and only then reassemble or write parent outputs.
9. Test enabled and disabled modes against the same correctness oracle.

Do not introduce buffering across `process_batch()` calls to improve the
duration distribution. That changes latency, failure, ownership, and end-of-
stream behavior and would require a separate queue/flush design.

### Model-specific cautions

- **ASR:** the current contract concatenates independently decoded segment
  text in temporal order.
- **SED:** preserve frame arrays, valid-frame counts, timestamps, and sidecar
  paths when scattering. Splitting also requires frame-offset reconstruction.
- **VAD:** independent splits can change onset/offset decisions and merged
  speech regions.
- **Diarization:** speaker clustering and identity may depend on the whole
  recording; arbitrary segmentation is not equivalent.
- **Alignment:** flattened segment metadata must remain paired with the right
  hypothesis after reordered calls.
- **File-producing stages:** derive paths from saved parent metadata, never
  from reordered call position.

Reordering independent whole inputs is often safe. Segmentation and stitching
always require a model-specific correctness design.

## Testing and parity

Unit tests should use a recording adapter stub and cover:

- missing, Boolean, negative, `NaN`, and infinite budgets, plus model limits
  shorter than one sample;
- an empty input and zero-duration audio;
- exact budget fills and a candidate that begins the next call;
- original-order and stable duration-order call membership;
- equal-duration stability;
- the `[1, 2, 2]` budget-`4` case that distinguishes the optimal boundary
  plan from sorted greedy packing;
- planning across segments from multiple rows in one call, but never across
  two `process_batch()` calls;
- exact model-duration boundaries, multi-segment parents, and every final
  remainder;
- result scatter, parent stitching, skip/error paths, and wrong result counts;
- proof that `batch_size` does not impose an adapter item-count cap.

For a local parity run, use a fixed cohort with short, medium, boundary-equal,
long, and segmented-remainder audio. Hold model settings, software, hardware,
input order, and candidate windows constant. First verify row IDs, row counts,
skip classifications, transcript equality or the documented tolerance, and
exact-once segment coverage. Only then compare padding efficiency, throughput,
latency, RAM, and VRAM. Local bucketing guarantees the planning contract, not
a performance win for every workload.

## Explicit non-goals

This design intentionally does not provide:

- global, partition-wide, or cross-worker bucketing;
- planning across multiple `process_batch()` calls;
- queues, timers, flush hooks, or end-of-stream state;
- static bucket edges or per-bucket configuration;
- separate adapter or per-duration-group item-count controls;
- feature-weighted cost estimation;
- reordering beyond the stable duration sort or optimization across
  noncontiguous subsets;
- adaptive or OOM-retry scheduling;
- automatic budget tuning;
- a guarantee of higher throughput for every model and distribution.

For `N` segments, planning is `O(N)` with bucketing off. With bucketing on,
stable sorting costs `O(N log N)` and the exact boundary dynamic program costs
`O(N^2)`, for `O(N^2)` total time and `O(N)` auxiliary storage. Given the same
ordered prepared segments and configuration, the plan is deterministic.

## Source and test map

- [`asr/stage.py`](asr/stage.py): waveform preparation, unconditional
  segmentation, local planning, adapter execution, scatter, and stitching.
- [`model_input_segmentation.py`](../model_input_segmentation.py): contiguous
  model-safe segment planning and validation.
- [`inference/base.py`](base.py): shared adapter lifecycle and input handling;
  deliberately not a generic batching planner.
- [`models/asr/base.py`](../../../models/asr/base.py): ordered ASR adapter input
  and result contract.
- [`test_asr_stage.py`](../../../../tests/stages/audio/inference/test_asr_stage.py):
  actor budgets, mode behavior, local scope, segmentation, ordering, and error
  coverage.
- [`test_model_input_segmentation.py`](../../../../tests/stages/audio/test_model_input_segmentation.py):
  exact boundaries, remainders, one-sample limits, and zero-length inputs.
- [FastConformer tutorial](../../../../tutorials/audio/nemo_fastconformer/README.md):
  runnable ASR configuration and usage.
