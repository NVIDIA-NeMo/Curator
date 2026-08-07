# Audio performance report parity: PR #2296 versus main

This document records the exact Qwen3-Omni transcription output and performance-report result from the live 2026-08-06 parity canary. It compares PR [#2296](https://github.com/NVIDIA-NeMo/Curator/pull/2296) with Curator main at the PR branch point. This is a correctness and report-lifecycle canary, not a statistically powered throughput benchmark.

## Result

The frozen acceptance criteria passed:

- both independent Kratos runs and all five tasks in each run succeeded;
- both arms returned eight unique, non-empty predictions for the same eight audio inputs;
- five predictions were exact matches, all eight normalized similarities were at least `0.90`, and mean normalized similarity was `0.9895596305990517`;
- the PR arm wrote one native `qwen_omni_performance.json` with schema version `1`, all five ordered pipeline stages, 19 unique invocation records, unique non-empty invocation IDs, and a positive `221.20097979158163` second pipeline wall time;
- main wrote no `qwen_omni_performance.json`, because that terminal report and its writer configuration do not exist at the reference commit;
- neither output prefix contained resampled WAV scratch files.

## Frozen experiment identity

| Item | Exact value |
| --- | --- |
| PR Curator commit | `56e6d8b5cde4d8beea0076dddd2e9e03949a13e9` |
| Main reference commit | `464e3fb7aebca70b9a510554637c2969434abe77` |
| Shared NvLLMOps commit | `19f0d55e36f5391b2865c704d1c2cfdc412d3fdd` |
| PR image | `nvcr.io/nvidian/tegra-audio/speech-mlops@sha256:e21b036a01e781319083f99e2d65ee782e27d476e436e3c8196d71555ffe11f0` |
| Main image | `nvcr.io/nvidian/tegra-audio/speech-mlops@sha256:bc5b6bfb03cd656e7feffd352dbb3b206d7d4bb1b2642256b722b961d582f25f` |
| Input manifest SHA-256 | `7b9a5708c2de96da8c8e8b6b253bc8fd17ce80fa2cff48fff57a8a9081e86707` |
| Rows / audio duration | `8` / `569.4863968253968` seconds |
| Model | `Qwen/Qwen3-Omni-30B-A3B-Instruct` |
| Executor / hardware | `RayDataExecutor`; one node; two GPUs |
| PR Kratos run | [`1a1c95f8-15ac-42fe-ab57-fa6b8ee43b20`](https://xp-rivanemo.kratos.nvidia.com/_/pipeline/?ns=rivanemo-asr#/runs/details/1a1c95f8-15ac-42fe-ab57-fa6b8ee43b20) |
| Main Kratos run | [`be85d06f-d865-42fe-b489-82a5612dbb39`](https://xp-rivanemo.kratos.nvidia.com/_/pipeline/?ns=rivanemo-asr#/runs/details/be85d06f-d865-42fe-b489-82a5612dbb39) |

The main image was reused from an earlier exact-pair build because its embedded Curator and NvLLMOps commits were already immutable and registry-verified. A reference image does not need to be rebuilt when that source pair is unchanged.

The only branch-native configuration difference was the PR writer setting:

```yaml
performance_report_path: ${output_dir}/qwen_omni_performance.json
```

The input, prompt, model, batch size (`32`), maximum output tokens (`256`), sampling (`temperature: 0.0`, `top_k: 1`), executor, NvLLMOps harness, and hardware request were controlled across both arms.

## Exact comparison summary

```json
{
  "average_normalized_similarity": 0.9895596305990517,
  "normalized_exact_matches": 5,
  "normalized_similarity_ge_0_90": 8,
  "normalized_similarity_ge_0_95": 7,
  "raw_exact_matches": 5,
  "reference_empty_predictions": [],
  "reference_only_identities": [],
  "reference_rows": 8,
  "shared_identities": 8,
  "target_empty_predictions": [],
  "target_only_identities": [],
  "target_rows": 8
}
```

## Exact captured artifacts

The authoritative, unmodified files collected from Swift are stored beside this README:

| Artifact | SHA-256 |
| --- | --- |
| [PR native output](parity_pr2296/pr_output.jsonl) | `d6bf75394d5b77cc413f734cf4d5c1df39998a14241f1914b7d9366548dcabd3` |
| [Main native output](parity_pr2296/main_output.jsonl) | `3ac414a20d5d7bbd3d2a4ef2f7d9cc2419536b2b0866d32eb28b508f417178e3` |
| [PR terminal performance report](parity_pr2296/pr_performance.json) | `9e8c676b5a56fbc0a2527202edbfd3077d601bb57d897ae11ca6c43a8a12d8b4` |
| [Complete row-by-row comparison](parity_pr2296/comparison.json) | `ed10610e007d952c737d4111ae50065d978f3bcd3052c7cf3789ff3c01577612` |

There is deliberately no main performance-report file: the exact Swift inventory contained zero `qwen_omni_performance.json` objects for main and one for the PR arm.

## Performance-report delta

Main exposed performance only through legacy returned-task records: each of the eight result tasks carried five records, and the harness deduplicated them to 19 unique records. It had no public pipeline-owned performance sink, stable stage IDs, invocation IDs, pipeline wall time, or native terminal report.

The PR arm moved those same 19 unique invocation records to `pipeline.performance_records`; every returned task carried zero performance records. The terminal writer serialized them once. The exact ordered stages were:

1. `000:file_partitioning` — 1 invocation
2. `001:manifest_reader_stage` — 1 invocation
3. `002:ResampleAudio` — 8 invocations
4. `003:ASR_inference` — 1 invocation for all 8 items, with `input_data_size_mb: 0.0033893585205078125`
5. `004:manifest_writer` — 8 invocations

The report contains no GPU identity or utilization fields from extended backend telemetry; those are intentionally outside PR #2296.

## Human-readable row comparison

The exact artifact files above contain all fields and all eight complete predictions. The table gives the complete row inventory; representative `pred_text` values are reproduced verbatim below.

| Input identity | Normalized similarity | Raw exact match |
| --- | ---: | --- |
| `0pRq-1mXNwE.opus` | `0.9937106918238994` | no |
| `3OhKrjv1U9I.m4a` | `1.0` | yes |
| `3sTKAKt905E.opus` | `0.9956140350877193` | no |
| `DQCmrkeRayk.opus` | `1.0` | yes |
| `VRDKD-HGS4k.opus` | `1.0` | yes |
| `fq2hedfn69s.opus` | `0.9271523178807947` | no |
| `ooSDkEWptyw.opus` | `1.0` | yes |
| `wYoirH7IhHQ.m4a` | `1.0` | yes |

Selected `pred_text` values below preserve the emitted words and punctuation; Markdown wrapping is presentational. The exact JSON serialization for all rows is in the linked artifacts. The complete PR `output.jsonl` SHA-256 is `d6bf75394d5b77cc413f734cf4d5c1df39998a14241f1914b7d9366548dcabd3`; the complete main `output.jsonl` SHA-256 is `3ac414a20d5d7bbd3d2a4ef2f7d9cc2419536b2b0866d32eb28b508f417178e3`.

### `DQCmrkeRayk.opus` — similarity `1.0`

PR #2296 and main both emitted:

> Our instant index tonight starts with a blooper reel from a galaxy far far away never before seen outtakes from Star Wars Return of the Jedi. Chewbacca and Han Solo keep your eye on Harrison Ford's headset here and Luke Skywalker breaking character listen to this. Now let's get some distance before that thing goes supernova. How do you pronounce supernova with inflection supernova or supernova. That's right they were asking how do you pronounce supernova and look who wandered into this drugstore a flock of ducks thirty of them in fact waddling through the aisles of a CVS in New York City. They wouldn't leave so quick thinking customers left a trail of popcorn we're told. It worked. The ducks waddled back out to the street guess they were looking for a little snack there. As they say only in New York. And you'll remember the Ohio State marching band that thrilled us with their own version of Thriller. Well they are back at it tonight paying tribute to Hollywood's first man of steel Superman then the boy wizard Harry Potter and finally a T-Rex. That's a little nod to Jurassic Park.

### `ooSDkEWptyw.opus` — similarity `1.0`

PR #2296 and main both emitted:

> I went on a date with a girl from Los Feliz and at the end of the date I turned I go in to kiss and she went the cobra and I you got cobraed I got cobraed you got cobraed the cobra whoa dude so I'm like I had a great night she goes yeah me too we should do it again I go yeah cobraed cobraed

### `0pRq-1mXNwE.opus` — similarity `0.9937106918238994`

PR #2296:

> You wanna cough drop even, you wanna cough drop? Take it. No! Whoa! What the fuck? Poor Addison Rae just swung across the wall and hit the after dark wall. Oh my god, she's got a concussion I bet. Oh you mean a better believe I believe. You said it dude. I think I've been this drunk my life. I love when someone gets drunk and becomes a wrecking ball. I am just straight up Miley Cyrus wrecking.

Main:

> You wanna cough drop even, you wanna cough drop? Take it. No! Whoa! What the fuck? Poor Addison Rae just swung across the wall and hit the after dark wall. Oh my god, she's got a concussion I bet. Oh you mean a better believe I believe. You said it dude. I think I've been drunk my life. I love when someone gets drunk and becomes a wrecking ball. I am just straight up Miley Cyrus wrecking.

### `3OhKrjv1U9I.m4a` — similarity `1.0`

PR #2296 and main both emitted:

> Summer is over, which for many of us means back to work. Here are three things you can do to get back on track. First, tackle small goals before anything else. If you jump right into the unread emails and unfinished projects, you're in for a shock, making you feel overwhelmed rather than productive. The to-do list is your best friend here. Break down your projects into manageable, achievable tasks. If you start the day crossing things off, you'll feel accomplished and motivated to keep going. Second, start to get back into your usual routine. Over the summer, you may have found yourself staying up later than usual, rolling into work a little late or eating more junk food. Getting back into your typical routine, whether it's a set bedtime or enjoying your morning coffee, will help signal your body and mind that things are back to normal and set a good tone for your workday. Finally, keep the summer vibes going, within reason. The sense of renewed energy from the summer feeling can actually help you be a more productive worker. For example, plan an alfresco lunch with a coworker. You'll get to enjoy the late summer sun and a change of scenery during your workday. Or, for when you're back at your desk, print

### `3sTKAKt905E.opus` — similarity `0.9956140350877193`

PR #2296:

> President Obama has invited his successor to the White House today in one of the first steps to aim for a peaceful transition of power. So the president and Donald Trump have been openly and bitterly critical of each other, but now they're both calling for unity. Here's the president and Vice President Biden meeting with staffers yesterday to talk about the election, telling them to keep their heads up.
> ABC's Arlette Science has more from Washington. Arlette, good morning. Diana and Kendis, it's a moment many in this country never imagined. President Obama welcoming Donald Trump to the Oval Office as the next president.

> The White House admits this won't be an easy meeting. It's well documented President Obama thinks Donald Trump is unfit to be commander in chief, while Donald Trump has promised to undo much of President Obama's legacy.
> But today President Obama will put aside those differences. Here he was in the Rose Garden yesterday. We are now all rooting for his success in uniting and leading the country.

> The peaceful transition of power is one of the hallmarks of our democracy. And over the next few months, we are going to show that to the world.
> Now part of the transition involves intelligence briefings. As president-elect, Donald Trump can now receive the exact same daily

Main:

> President Obama has invited his successor to the White House today in one of the first steps to aim for a peaceful transition of power. So the president and Donald Trump have been openly and bitterly critical of each other, but now they're both calling for unity.

> Here's the president and Vice President Biden meeting with staffers yesterday to talk about the election, telling them to keep their heads up. ABC's Arlette Science has more from Washington. Arlette, good morning.
> Diane and Kendis, it's a moment many in this country never imagined. President Obama welcoming Donald Trump to the Oval Office as the next president. The White House admits this won't be an easy meeting.
> It's well documented President Obama thinks Donald Trump is unfit to be commander in chief, while Donald Trump has promised to undo much of President Obama's legacy.

> But today President Obama will put aside those differences. Here he was in the Rose Garden yesterday. We are now all rooting for his success in uniting and leading the country.
> The peaceful transition of power is one of the hallmarks of our democracy. And over the next few months, we are going to show that to the world.
> Now part of the transition involves intelligence briefings. As president-elect, Donald Trump can now receive the exact same daily

### `wYoirH7IhHQ.m4a` — similarity `1.0`

PR #2296 and main both emitted:

> Mr. President, Mr. President, and I just talked to him so I know he's watching. Thank you so much for everything, for the opportunities, for coming to to Mississippi yesterday. Two stops, phenomenal. Tupelo was unbelievable, the coast was unbelievable. We got to ride together and I sure appreciated the time that he gave me. Senator Lindsey, Vice President Mike Pence, my guys, they turned out for me. My colleagues have supported me. Right now, Mr. President, thank you so much for all of your help. I'm so humbled for the honor that you have given me to elect me as your United States Senator. It is hard to describe the feelings tonight for me and my family. But tonight in this victory, the reason we won is because Mississippians know me and they know my heart and thank you for stepping up, Mississippians. I've said all along, this isn't about me. This is about the people of Mississippi and what's important to the people of Mississippi. You know, this this win tonight, this victory, it's about our conservative values. It's about the things that mean the most to all of us Mississippians, our faith, our family. But it's those things that I will take to
