#!/usr/bin/env python3
"""
Build a JSON manifest of (session_id, sample_rate, wav paths, rttm paths) pairs
from the DavidAI_Micro dataset layout.

Usage:
    python build_file_list.py \
        --data_dir /disk_a_nvd/datasets/sub4spk_finetune_datasets/DavidAI_Micro \
        --rttm_dir /disk_a_nvd/datasets/sub4spk_finetune_datasets/DavidAI_Micro/tjp_annotated_rttms \
        --output   file_list.json \
        [--sample_rate 16000]
"""
import argparse
import json
import os
import re
from pathlib import Path


def discover_sessions(rttm_dir: str) -> dict[str, dict[str, str]]:
    """Return {session_id: {speaker_label: rttm_path}} from the RTTM folder."""
    pattern = re.compile(r"^(.+)-(speaker\d+)\.rttm$")
    sessions: dict[str, dict[str, str]] = {}
    for fname in sorted(os.listdir(rttm_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        sid, spk = m.group(1), m.group(2)
        sessions.setdefault(sid, {})[spk] = os.path.join(rttm_dir, fname)
    return sessions


def build_manifest(data_dir: str, rttm_dir: str, sample_rate: int = 16000) -> list[dict]:
    sessions = discover_sessions(rttm_dir)
    manifest = []
    for sid in sorted(sessions):
        session_dir = os.path.join(data_dir, sid)
        if not os.path.isdir(session_dir):
            print(f"WARNING: session dir not found, skipping: {session_dir}")
            continue

        mixed_wav = os.path.join(session_dir, f"{sid}.wav")
        channels = {}
        for spk in sorted(sessions[sid]):
            ch_wav = os.path.join(session_dir, f"{spk}.wav")
            if os.path.isfile(ch_wav):
                channels[spk] = ch_wav
            else:
                print(f"WARNING: channel wav missing: {ch_wav}")

        entry = {
            "session_id": sid,
            "sample_rate": sample_rate,
            "mixed_wav": mixed_wav if os.path.isfile(mixed_wav) else None,
            "channels": channels,
            "rttms": sessions[sid],
        }
        manifest.append(entry)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build QA file-list manifest")
    parser.add_argument(
        "--data_dir",
        default="/disk_a_nvd/datasets/sub4spk_finetune_datasets/DavidAI_Micro",
    )
    parser.add_argument(
        "--rttm_dir",
        default="/disk_a_nvd/datasets/sub4spk_finetune_datasets/DavidAI_Micro/tjp_annotated_rttms",
    )
    parser.add_argument("--output", default="file_list.json")
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Hz for all WAVs in each session (written as sample_rate in the manifest)",
    )
    args = parser.parse_args()

    manifest = build_manifest(args.data_dir, args.rttm_dir, args.sample_rate)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest)} sessions to {args.output}")


if __name__ == "__main__":
    main()
