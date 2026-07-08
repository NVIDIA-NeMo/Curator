# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert NFA word-level CTM files to Praat TextGrid format."""

import json
import sys
from pathlib import Path


def parse_ctm(ctm_path: Path) -> list[tuple[float, float, str]]:
    """Parse a CTM file into a list of (start, end, word) tuples."""
    words = []
    with ctm_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            start = float(parts[2])
            duration = float(parts[3])
            word = parts[4]
            words.append((start, start + duration, word))
    return words


def write_textgrid(words: list[tuple[float, float, str]], output_path: Path, audio_duration: float | None = None):
    """Write a TextGrid file with a single 'words' tier."""
    if not words:
        return

    xmin = 0.0
    xmax = audio_duration if audio_duration else words[-1][1] + 0.01

    intervals = []
    prev_end = 0.0
    for start, end, word in words:
        if start > prev_end + 0.001:
            intervals.append((prev_end, start, ""))
        intervals.append((start, end, word))
        prev_end = end

    if prev_end < xmax:
        intervals.append((prev_end, xmax, ""))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write('File type = "ooTextFile"\n')
        f.write('Object class = "TextGrid"\n\n')
        f.write(f"xmin = {xmin}\n")
        f.write(f"xmax = {xmax}\n")
        f.write("tiers? <exists>\n")
        f.write("size = 1\n")
        f.write("item []:\n")
        f.write("    item [1]:\n")
        f.write('        class = "IntervalTier"\n')
        f.write('        name = "words"\n')
        f.write(f"        xmin = {xmin}\n")
        f.write(f"        xmax = {xmax}\n")
        f.write(f"        intervals: size = {len(intervals)}\n")
        for i, (s, e, text) in enumerate(intervals, 1):
            f.write(f"        intervals [{i}]:\n")
            f.write(f"            xmin = {s}\n")
            f.write(f"            xmax = {e}\n")
            f.write(f'            text = "{text}"\n')


def convert_all(ctm_dir: Path, textgrid_dir: Path) -> dict[str, str]:
    """Convert all CTM files in a directory to TextGrid files.

    Returns a dict mapping utterance_id -> textgrid path.
    """
    textgrid_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    ctm_files = sorted(ctm_dir.glob("*.ctm"))
    total = len(ctm_files)
    print(f"  Found {total} CTM files to convert")

    for i, ctm_path in enumerate(ctm_files):
        utt_id = ctm_path.stem
        words = parse_ctm(ctm_path)
        if not words:
            continue

        tg_path = textgrid_dir / f"{utt_id}.TextGrid"
        write_textgrid(words, tg_path)
        index[utt_id] = str(tg_path)

        if (i + 1) % 50000 == 0:
            print(f"  Converted {i + 1}/{total} ...")

    print(f"  Done: {len(index)} TextGrid files written")
    return index


def main():
    ctm_dir = Path("mls_workdir/mls/nfa_output/french_train/nfa_work/ctm/words")
    tg_dir = Path("mls_workdir/mls/nfa_output/french_train/textgrids")
    s7_path = Path("mls_workdir/mls/pipeline_state/french_train/stage7_tasks.jsonl")
    s9_path = Path("mls_workdir/mls/pipeline_state/french_train/stage9_nfa_tasks.jsonl")

    print("Step 1: Converting CTM -> TextGrid ...")
    index = convert_all(ctm_dir, tg_dir)

    print("\nStep 2: Writing stage9_nfa_tasks.jsonl ...")
    tasks = []
    with s7_path.open() as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))

    matched = 0
    skipped = 0
    with s9_path.open("w") as fout:
        for task in tasks:
            task_id = task["task_id"]
            tg_path = index.get(task_id, "")
            if tg_path:
                task["data"]["textgrid_filepath"] = tg_path
                matched += 1
            else:
                task["data"]["nfa_skipped"] = True
                skipped += 1
            fout.write(json.dumps(task, ensure_ascii=False) + "\n")

    print(f"  Matched: {matched}, Skipped: {skipped}")
    print(f"  Wrote: {s9_path}")


if __name__ == "__main__":
    main()
