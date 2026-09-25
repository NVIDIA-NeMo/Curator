# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
BUILD_CORPUS = REPO_ROOT / ".cursor/scripts/curator-pr-review/build_corpus.py"
AUDIO_REVIEW = REPO_ROOT / ".cursor/skills/review-curator-audio-pr/scripts/review_audio_pr.sh"
AUDIO_PATHS = REPO_ROOT / ".cursor/skills/review-curator-audio-pr/audio_paths.sh"
PULL_AUDIO_CORPUS = REPO_ROOT / ".cursor/skills/review-curator-audio-pr/scripts/pull_audio_pr_corpus.sh"
TODAY = "2026-08-17"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _audio_numbers_file(since: int = 1608) -> str:
    result = subprocess.run(  # noqa: S603 - fixed shell helper under test
        [
            "/usr/bin/bash",
            "-c",
            'source "$1"; audio_corpus_numbers_file "$2"',
            "audio-corpus-scope",
            str(AUDIO_PATHS),
            str(since),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _seed_corpus_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True)
    (cache_dir / ".corpus.lock").touch()
    for numbers_file in ("_audio_pr_numbers.txt", _audio_numbers_file()):
        (cache_dir / numbers_file).write_text("1701\n1702\n", encoding="utf-8")
    for number, sentinel in ((1701, "first-cache-sentinel"), (1702, "second-cache-sentinel")):
        _write_json(
            cache_dir / f"pr{number}_gh.json",
            {
                "author": {"login": "author"},
                "createdAt": "2026-01-01T00:00:00Z",
                "state": "MERGED",
                "title": f"Audio PR {number}",
                "url": f"https://github.com/NVIDIA-NeMo/Curator/pull/{number}",
            },
        )
        _write_json(cache_dir / f"pr{number}_reviews.json", [])
        _write_json(
            cache_dir / f"pr{number}_review_comments.json",
            [
                {
                    "body": sentinel,
                    "html_url": f"https://github.com/NVIDIA-NeMo/Curator/pull/{number}#discussion",
                    "line": 12,
                    "path": "nemo_curator/stages/audio/example.py",
                    "user": {"login": "reviewer"},
                }
            ],
        )
        _write_json(cache_dir / f"pr{number}_issue_comments.json", [])


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - commands are fixed test-owned scripts
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _assert_render_only(review_dir: Path) -> Path:
    files = list(review_dir.iterdir())
    assert [path.name for path in files] == [f"audio_pr_corpus_{TODAY.replace('-', '_')}.md"]
    assert not list(review_dir.glob("pr*.json"))
    return files[0]


def _make_fake_gh(tmp_path: Path) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    call_log = tmp_path / "gh-calls.log"
    fake_gh = bin_dir / "gh"
    fake_gh.write_text(
        f"""#!{sys.executable}
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
with Path(os.environ["FAKE_GH_CALL_LOG"]).open("a", encoding="utf-8") as log:
    log.write(" ".join(args) + "\\n")

if args[0] == "api":
    endpoint = args[-1]
    if os.environ.get("FAKE_GH_FAIL_ENDPOINT", "") in endpoint and os.environ.get("FAKE_GH_FAIL_ENDPOINT"):
        raise SystemExit(23)
    updated_at = os.environ.get("FAKE_GH_UPDATED_AT", "2026-08-17T00:00:00Z")
    if "/pulls?state=all" in endpoint:
        print(json.dumps({{"number": 1701, "updated_at": updated_at}}))
    elif endpoint.endswith("/pulls/1701/files"):
        print(json.dumps({{"filename": "nemo_curator/stages/audio/example.py"}}))
    elif endpoint.endswith("/pulls/1701/comments"):
        print(json.dumps({{"body": os.environ.get("FAKE_GH_COMMENT_BODY", "cached comment")}}))
elif args[:2] == ["pr", "view"]:
    print(json.dumps({{
        "author": {{"login": "author"}},
        "createdAt": "2026-08-01T00:00:00Z",
        "updatedAt": os.environ.get("FAKE_GH_UPDATED_AT", "2026-08-17T00:00:00Z"),
        "number": 1701,
        "state": "OPEN",
        "title": "Audio PR",
        "url": "https://github.com/NVIDIA-NeMo/Curator/pull/1701"
    }}))
else:
    raise SystemExit(f"unsupported fake gh command: {{args}}")
""",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)
    return bin_dir, call_log


def test_build_corpus_reads_shared_cache_and_writes_only_rendered_markdown(tmp_path: Path) -> None:
    cache_dir = tmp_path / "shared-cache"
    review_dir = tmp_path / "review-a"
    _seed_corpus_cache(cache_dir)

    result = _run(
        [
            sys.executable,
            str(BUILD_CORPUS),
            "--cache-dir",
            str(cache_dir),
            "--outdir",
            str(review_dir),
            "--numbers-file",
            "_audio_pr_numbers.txt",
            "--today",
            TODAY,
            "--output-prefix",
            "audio_pr_corpus",
        ],
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    rendered = _assert_render_only(review_dir)
    text = rendered.read_text(encoding="utf-8")
    assert "first-cache-sentinel" in text
    assert "second-cache-sentinel" in text


def test_build_corpus_rejects_overlapping_cache_and_output(tmp_path: Path) -> None:
    cache_dir = tmp_path / "shared-cache"
    _seed_corpus_cache(cache_dir)
    result = _run(
        [
            sys.executable,
            str(BUILD_CORPUS),
            "--cache-dir",
            str(cache_dir),
            "--outdir",
            str(cache_dir / "rendered"),
            "--numbers-file",
            "_audio_pr_numbers.txt",
        ],
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert "must not overlap" in result.stderr
    assert not (cache_dir / "rendered").exists()


def test_audio_wrapper_reuses_one_cache_for_two_review_directories(tmp_path: Path) -> None:
    cache_root = tmp_path / "workspace-cache"
    cache_dir = cache_root / "NVIDIA-NeMo_Curator/audio-corpus"
    _seed_corpus_cache(cache_dir)
    before = {path.relative_to(cache_dir): path.read_bytes() for path in cache_dir.iterdir()}

    env = os.environ.copy()
    env["CURATOR_PR_REVIEW_CACHE_ROOT"] = str(cache_root)
    for name in ("review-a", "review-b"):
        workdir = tmp_path / name
        workdir.mkdir()
        result = _run(
            [str(AUDIO_REVIEW), "build-corpus", "--outdir", str(workdir), "--today", TODAY],
            cwd=workdir,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        _assert_render_only(workdir)

    after = {path.relative_to(cache_dir): path.read_bytes() for path in cache_dir.iterdir()}
    assert after == before


def test_corpus_puller_rejects_per_review_outdir(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    result = _run([str(PULL_AUDIO_CORPUS), "--outdir", str(review_dir)], cwd=tmp_path)

    assert result.returncode == 2
    assert "replaced by --cache-dir" in result.stderr
    assert not review_dir.exists()


def test_corpus_puller_reuses_shared_cache_across_review_directories(tmp_path: Path) -> None:
    cache_dir = tmp_path / "shared-cache"
    review_a = tmp_path / "review-a"
    review_b = tmp_path / "review-b"
    review_a.mkdir()
    review_b.mkdir()
    bin_dir, call_log = _make_fake_gh(tmp_path)
    env = os.environ.copy()
    env["FAKE_GH_CALL_LOG"] = str(call_log)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    command = [str(PULL_AUDIO_CORPUS), "--since", "1608", "--cache-dir", str(cache_dir)]

    first = _run(command, cwd=review_a, env=env)
    assert first.returncode == 0, first.stderr
    assert (cache_dir / "pr1701_review_comments.json").is_file()
    assert not list(review_a.iterdir())

    call_log.write_text("", encoding="utf-8")
    second = _run(command, cwd=review_b, env=env)
    assert second.returncode == 0, second.stderr
    assert not list(review_b.iterdir())
    second_calls = call_log.read_text(encoding="utf-8")
    assert "/pulls?state=all" in second_calls
    assert "/pulls/1701/files" not in second_calls
    assert "/pulls/1701/reviews" not in second_calls
    assert "/pulls/1701/comments" not in second_calls
    assert "pr view 1701" not in second_calls


def test_failed_refresh_preserves_last_complete_corpus(tmp_path: Path) -> None:
    cache_dir = tmp_path / "shared-cache"
    review_dir = tmp_path / "review"
    bin_dir, call_log = _make_fake_gh(tmp_path)
    env = os.environ.copy()
    env["FAKE_GH_CALL_LOG"] = str(call_log)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    pull_command = [str(PULL_AUDIO_CORPUS), "--since", "1608", "--cache-dir", str(cache_dir)]

    first = _run(pull_command, cwd=tmp_path, env=env)
    assert first.returncode == 0, first.stderr
    preserved_files = [
        cache_dir / _audio_numbers_file(),
        cache_dir / "pr1701_gh.json",
        cache_dir / "pr1701_reviews.json",
        cache_dir / "pr1701_review_comments.json",
        cache_dir / "pr1701_issue_comments.json",
    ]
    before = {path.name: path.read_bytes() for path in preserved_files}

    env["FAKE_GH_UPDATED_AT"] = "2026-08-18T00:00:00Z"
    env["FAKE_GH_COMMENT_BODY"] = "must-not-be-published"
    env["FAKE_GH_FAIL_ENDPOINT"] = "/pulls/1701/comments"
    failed = _run(pull_command, cwd=tmp_path, env=env)
    assert failed.returncode != 0
    assert {path.name: path.read_bytes() for path in preserved_files} == before
    assert not list(cache_dir.glob("*.tmp.*"))

    rendered = _run(
        [
            str(AUDIO_REVIEW),
            "build-corpus",
            "--cache-dir",
            str(cache_dir),
            "--outdir",
            str(review_dir),
            "--today",
            TODAY,
        ],
        cwd=tmp_path,
    )
    assert rendered.returncode == 0, rendered.stderr
    corpus_text = _assert_render_only(review_dir).read_text(encoding="utf-8")
    assert "cached comment" in corpus_text
    assert "must-not-be-published" not in corpus_text


def test_corpus_selection_is_isolated_by_since_scope(tmp_path: Path) -> None:
    cache_dir = tmp_path / "shared-cache"
    bin_dir, call_log = _make_fake_gh(tmp_path)
    env = os.environ.copy()
    env["FAKE_GH_CALL_LOG"] = str(call_log)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    for since in (1608, 1701):
        result = _run(
            [str(PULL_AUDIO_CORPUS), "--since", str(since), "--cache-dir", str(cache_dir)],
            cwd=tmp_path,
            env=env,
        )
        assert result.returncode == 0, result.stderr

    broad_manifest = cache_dir / _audio_numbers_file(1608)
    narrow_manifest = cache_dir / _audio_numbers_file(1701)
    assert broad_manifest != narrow_manifest
    assert broad_manifest.read_text(encoding="utf-8").split() == ["1701"]
    assert narrow_manifest.read_text(encoding="utf-8").split() == []

    broad_review = tmp_path / "broad-review"
    narrow_review = tmp_path / "narrow-review"
    for since, review_dir in ((1608, broad_review), (1701, narrow_review)):
        result = _run(
            [
                str(AUDIO_REVIEW),
                "build-corpus",
                "--since",
                str(since),
                "--cache-dir",
                str(cache_dir),
                "--outdir",
                str(review_dir),
                "--today",
                TODAY,
            ],
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr

    assert "PRs in corpus: **1**" in _assert_render_only(broad_review).read_text(encoding="utf-8")
    assert "PRs in corpus: **0**" in _assert_render_only(narrow_review).read_text(encoding="utf-8")


def test_corpus_puller_waits_for_concurrent_refresh(tmp_path: Path) -> None:
    cache_dir = tmp_path / "shared-cache"
    cache_dir.mkdir()
    bin_dir, call_log = _make_fake_gh(tmp_path)
    env = os.environ.copy()
    env["FAKE_GH_CALL_LOG"] = str(call_log)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    lock_file = (cache_dir / ".corpus.lock").open("a+", encoding="utf-8")
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    process = subprocess.Popen(  # noqa: S603 - command is a fixed test-owned script
        [str(PULL_AUDIO_CORPUS), "--since", "1608", "--cache-dir", str(cache_dir)],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=0.2)
        assert not call_log.exists()
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()

    _, stderr = process.communicate(timeout=5)
    assert process.returncode == 0, stderr


def test_build_corpus_waits_for_refresh_before_reading(tmp_path: Path) -> None:
    cache_dir = tmp_path / "shared-cache"
    review_dir = tmp_path / "review"
    _seed_corpus_cache(cache_dir)
    command = [
        sys.executable,
        str(BUILD_CORPUS),
        "--cache-dir",
        str(cache_dir),
        "--outdir",
        str(review_dir),
        "--numbers-file",
        "_audio_pr_numbers.txt",
        "--today",
        TODAY,
        "--output-prefix",
        "audio_pr_corpus",
    ]

    lock_file = (cache_dir / ".corpus.lock").open("a+", encoding="utf-8")
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    process = subprocess.Popen(  # noqa: S603 - command is a fixed test-owned script
        command,
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=0.2)
        assert not review_dir.exists()
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()

    _, stderr = process.communicate(timeout=5)
    assert process.returncode == 0, stderr
    _assert_render_only(review_dir)
