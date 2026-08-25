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

"""Unit tests for the deterministic safety guardrails (nemo_curator.audio_agent._safety)."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pytest

from nemo_curator.audio_agent import _safety
from nemo_curator.audio_agent.recipe import Recipe


class TestRedact:
    def test_strips_secrets_and_transcripts(self) -> None:
        obj = {
            "api_key": "sk-123",
            "hf_token": "hf_x",
            "text": "hello world",
            "score": 3.5,
            "nested": {"password": "p", "keep": 1},
            "list": [{"secret": "s"}],
        }
        out = _safety.redact(obj)
        assert out["api_key"] == "<redacted-secret>"
        assert out["hf_token"] == "<redacted-secret>"  # noqa: S105
        assert out["nested"]["password"] == "<redacted-secret>"  # noqa: S105
        assert out["nested"]["keep"] == 1
        assert out["list"][0]["secret"] == "<redacted-secret>"  # noqa: S105
        assert out["text"].startswith("<redacted-transcript:")
        assert out["score"] == 3.5

    def test_can_keep_transcripts(self) -> None:
        assert _safety.redact({"text": "hi"}, redact_transcripts=False)["text"] == "hi"

    @pytest.mark.parametrize(
        "key",
        ["apiToken", "authToken", "accessToken", "APIToken", "passwords", "secretkey", "accesskey"],
    )
    def test_secret_keys_without_a_separator_are_still_redacted(self, key: str) -> None:
        """Matching on the key's WORDS is what keeps ``tokenizer_path`` intact, but the word
        boundary in ``apiToken`` is the case change alone -- lowercasing before splitting
        collapses it into one unknown word and the credential rides out to the host LLM in a
        stage's error payload, where no value-level pattern can catch an opaque token either.
        """
        assert _safety.redact({key: "9f3c1e-real-credential"})[key] == "<redacted-secret>"

    @pytest.mark.parametrize("key", ["tokenizer_path", "tokenizerPath", "audio_filepath_key", "score_key"])
    def test_case_splitting_does_not_start_redacting_ordinary_fields(self, key: str) -> None:
        """The other half of the same rule: a destroyed value is persisted into run records and
        compared during reuse, so over-redaction outlives the display it was meant to protect."""
        assert _safety.redact({key: "keep-me"})[key] == "keep-me"

    def test_strips_secret_values_embedded_in_error_strings(self) -> None:
        out = _safety.redact(
            {
                "reason": (
                    "credential=plain-value "
                    "HF_TOKEN=plain-token "
                    "AWS_ACCESS_KEY_ID=plain-access "
                    "Authorization: Bearer bearer-value"
                )
            }
        )
        reason = out["reason"]
        assert "plain-value" not in reason
        assert "plain-token" not in reason
        assert "plain-access" not in reason
        assert "bearer-value" not in reason
        assert reason.count("<redacted-secret>") == 4

    def test_strips_quoted_json_secret_assignments_and_multiword_values(self) -> None:
        redacted = _safety.redact_secret_text('{"api_key": "sk-demo-secret", "password": "two words secret"}')
        assert "sk-demo-secret" not in redacted
        assert "two words secret" not in redacted
        assert redacted.count("<redacted-secret>") == 2

    def test_strips_basic_auth_url_userinfo_and_jwt(self) -> None:
        basic = "dXNlcjpwYXNzd29yZA=="
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        redacted = _safety.redact_secret_text(
            f"Authorization: Basic {basic}; registry=https://alice:correct-horse@example.test/v2; assertion {jwt}"
        )
        assert basic not in redacted
        assert "alice:correct-horse" not in redacted
        assert jwt not in redacted
        assert "https://<redacted-secret>@example.test/v2" in redacted
        assert redacted.count("<redacted-secret>") == 3

    def test_strips_conservative_standalone_token_prefixes(self) -> None:
        # Assembled from split literals so these FAKE fixtures never appear as a
        # contiguous token in source (GitHub secret-scanning push protection matches the
        # xoxb-/ghp_/AKIA/sk-proj- prefixes). The runtime strings are unchanged, so the
        # redaction coverage is identical.
        secrets = (
            "sk-" + "proj-abcdefghijklmnopqrstuv",
            "ghp" + "_abcdefghijklmnopqrstuvwxyz123456",
            "AKIA" + "ABCDEFGHIJKLMNOP",
            "xoxb" + "-123456789012-abcdefghijklmnop",
        )
        redacted = _safety.redact_secret_text(" ".join(secrets))
        assert all(secret not in redacted for secret in secrets)
        assert redacted.count("<redacted-secret>") == len(secrets)

    def test_keeps_noncredential_basic_text_and_short_prefixed_terms(self) -> None:
        text = "basic authentication failed while importing sk-learn and checking ghp_status"
        assert _safety.redact_secret_text(text) == text


class TestWorkspaceLock:
    def test_off_by_default(self, monkeypatch) -> None:  # noqa: ANN001
        monkeypatch.delenv("AUDIO_AGENT_WORKSPACE", raising=False)
        assert _safety.workspace_root() is None
        assert _safety.path_violations(["/etc/passwd", "/anywhere/x.wav"]) == []

    def test_blocks_outside_allows_inside(self, monkeypatch, tmp_path) -> None:  # noqa: ANN001
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
        inside = str(tmp_path / "data" / "a.wav")
        violations = _safety.path_violations([inside, "/etc/passwd"])
        assert any("passwd" in v for v in violations)
        assert all("a.wav" not in v for v in violations)

    def test_allows_remote_uris(self, monkeypatch, tmp_path) -> None:  # noqa: ANN001
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
        assert _safety.path_violations(["s3://bucket/key", "http://h/x", None]) == []

    def test_file_uri_is_still_a_local_path(self, monkeypatch, tmp_path) -> None:  # noqa: ANN001
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
        inside = (tmp_path / "data.jsonl").as_uri()
        assert _safety.path_violations([inside]) == []
        assert any("passwd" in v for v in _safety.path_violations(["file:///etc/passwd"]))

    def test_local_uri_alias_is_still_a_local_path(self, monkeypatch, tmp_path) -> None:  # noqa: ANN001
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
        inside = f"local://{tmp_path}/data.jsonl"
        assert _safety.path_violations([inside]) == []
        assert any("passwd" in v for v in _safety.path_violations(["local:///etc/passwd"]))

    def test_a_file_uri_naming_another_host_is_not_a_workspace_path(self, monkeypatch, tmp_path) -> None:  # noqa: ANN001
        """``file://host/path`` names a path on *host*, not here. The lock rebuilds the ``//host``
        prefix before resolving precisely so that the remainder cannot be read as local.

        Drop those two lines -- they look like leftover URI fiddling -- and
        ``file://remotehost/<the workspace>/a.wav`` is accepted as a workspace path, because
        what is left is a string that starts with the root. ``localhost`` and an empty host do
        mean here, and must keep working.
        """
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
        inside = f"{tmp_path}/a.wav"

        assert _safety.path_violations([f"file://remotehost{inside}"]), (
            "another host's namespace was accepted as the local workspace"
        )
        assert _safety.path_violations([f"local://remotehost{inside}"])
        assert _safety.path_violations([f"file://localhost{inside}"]) == []
        assert _safety.path_violations([f"file://{inside}"]) == []

    @pytest.mark.parametrize("link_kind", ["directory", "file"])
    def test_a_symlink_out_of_the_workspace_does_not_escape_the_lock(
        self,
        monkeypatch,  # noqa: ANN001
        tmp_path,  # noqa: ANN001
        link_kind: str,
    ) -> None:
        """The lock is a string comparison against a resolved path, and the whole of the
        resolution is one ``os.path.realpath`` call. Swap it for ``abspath`` -- the reflex when
        a path looks like it just needs normalising -- and every path here reads as inside the
        workspace, because the escape lives in the filesystem rather than in the string.

        Nothing else in the suite distinguishes the two calls, so the lock could be opened by a
        refactor that stays green. A locked-down deployment is exactly where a planted link is
        worth planting.
        """
        workspace = tmp_path / "ws"
        outside = tmp_path / "outside"
        workspace.mkdir()
        outside.mkdir()
        (outside / "secret.txt").write_text("credentials")
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(workspace))

        if link_kind == "directory":
            (workspace / "escape").symlink_to(outside)
            candidate = str(workspace / "escape" / "secret.txt")
        else:
            (workspace / "secret.txt").symlink_to(outside / "secret.txt")
            candidate = str(workspace / "secret.txt")

        assert _safety.path_violations([candidate]), (
            f"a {link_kind} symlink pointing out of the workspace was accepted: {candidate}"
        )
        assert _safety.path_violations([str(workspace / "real.wav")]) == [], (
            "an ordinary path inside the workspace must still be allowed"
        )

    def test_recipe_path_params_flattens_list_valued_sources(self) -> None:
        stage = type("Stage", (), {"params": {"manifest_path": ["a.jsonl", "b.jsonl"]}})()
        recipe = type("Recipe", (), {"stages": [stage]})()
        assert _safety.recipe_path_params(recipe) == ["a.jsonl", "b.jsonl"]

    def test_recipe_path_params_ignores_semantic_path_key_fields(self) -> None:
        stage = type(
            "Stage",
            (),
            {
                "params": {
                    "audio_filepath_key": "audio_filepath",
                    "audio_path_resolution": "relative",
                    "split_filepaths_key": "split_filepaths",
                    "model_path": "nvidia/model-name",
                    "manifest_path": "inside.jsonl",
                }
            },
        )()
        recipe = type("Recipe", (), {"stages": [stage]})()
        assert _safety.recipe_path_params(recipe) == ["inside.jsonl"]


class TestRedactionDoesNotDefeatTheAgentsOwnWorkflow:
    def test_the_smoke_token_survives_the_word_rule_that_matches_it(self) -> None:
        """``smoke_token`` is evidence, not a credential.

        ``smoke`` produces it and ``run`` consumes it. Redacting it broke the one workflow it
        exists for: with AUDIO_AGENT_REQUIRE_SMOKE set, smoke returned ``<redacted-secret>``
        and no value the caller could pass would satisfy run, which refused every time.
        """
        out = _safety.redact({"smoke_token": "9f3c1e", "api_token": "9f3c1e"})

        assert out["smoke_token"] == "9f3c1e"  # noqa: S105
        assert out["api_token"] == "<redacted-secret>", "real credentials are still redacted"  # noqa: S105

    def test_a_transcript_is_redacted_whatever_container_holds_it(self) -> None:
        """Only bare strings were handled, so a transcript key holding a LIST -- per-segment
        or per-word text, the shape every segmenting stage produces -- fell through to the
        generic walk and reached the host LLM in full. The key was already known to be
        transcript-bearing; the container hid it.
        """
        out = _safety.redact({"text": ["first segment", "second segment"]})

        assert all("segment" not in str(item) for item in out["text"]), out["text"]
        assert all(str(item).startswith("<redacted-transcript:") for item in out["text"])


class TestRedactionWalksEveryContainerNotJustTheTwoNoticedFirst:
    """The walker handled ``dict``/``list``/``str`` and fell through on everything else,
    returning the original object untouched. A tuple or a set anywhere on the path meant the
    payload below it was never inspected -- so both rules (secret keys and transcripts) were
    silently skipped for whatever it held.

    This is reachable, not theoretical: ``verbs._examples_from_rows`` copies row values
    verbatim and ``verbs._jsonable`` admits tuples by name, so a manifest row carrying one
    lands in the ``examples`` payload that ``smoke``/``run`` hand to the host LLM.
    ``contracts._clean`` has always flattened tuples and sets, which is why this read as an
    oversight rather than a decision.
    """

    def test_a_secret_inside_a_tuple_is_redacted(self) -> None:
        out = _safety.redact({"data": ({"api_key": "9f3c1e-real-credential"},)})

        assert "9f3c1e-real-credential" not in json.dumps(out, default=str)
        assert out["data"][0]["api_key"] == "<redacted-secret>"

    def test_a_transcript_inside_a_tuple_is_redacted(self) -> None:
        """The list case was fixed once; the tuple carrying the same per-segment text walked
        straight past that fix."""
        out = _safety.redact({"text": ("first segment", "second segment")})

        assert "segment" not in json.dumps(out, default=str), out["text"]
        assert all(str(item).startswith("<redacted-transcript:") for item in out["text"])

    def test_a_transcript_inside_a_set_is_redacted(self) -> None:
        out = _safety.redact({"text": {"first segment", "second segment"}})

        assert "segment" not in json.dumps(out, default=str), out["text"]
        assert all(str(item).startswith("<redacted-transcript:") for item in out["text"])

    def test_a_secret_inside_a_set_is_redacted(self) -> None:
        """A set cannot hold a dict, so there is no key here for the key-based rule to match.
        What it can hold is text, and the inline rule only runs on members the walk reaches."""
        out = _safety.redact({"data": {"api_key=9f3c1e-real-credential"}})

        assert "9f3c1e-real-credential" not in json.dumps(out, default=str)
        assert out["data"] == ["api_key=<redacted-secret>"]

    def test_containers_are_flattened_to_json_serializable_lists(self) -> None:
        """Redacted output is about to be serialized, where neither type survives. Matching
        ``contracts._clean`` keeps the emitter from having to care which one it got."""
        out = _safety.redact({"a": (1, 2), "b": {3}})

        assert out["a"] == [1, 2]
        assert out["b"] == [3]
        json.dumps(out)  # must not raise: a bare set is not JSON-serializable

    def test_a_set_of_mixed_types_does_not_crash_the_walk(self) -> None:
        """Ordering a set needs a key; the natural ``sorted()`` raises on mixed types, and a
        redactor that throws is a redactor that gets bypassed."""
        out = _safety.redact({"mixed": {1, "two", None}})

        assert sorted(str(item) for item in out["mixed"]) == ["1", "None", "two"]

    def test_a_tuple_row_value_reaches_the_host_redacted(self) -> None:
        """End-to-end over the path that makes this reachable: a manifest row whose value is a
        tuple, through the same rows -> ``redact`` sequence smoke and run use."""
        from nemo_curator.audio_agent import verbs

        rows = [{"path": "/data/a.wav", "text": ("hello", "world"), "api_key": "9f3c1e-cred"}]
        out = _safety.redact(verbs._examples_from_rows(rows, limit=1))

        payload = json.dumps(out, default=str)
        assert "hello" not in payload, payload
        assert "9f3c1e-cred" not in payload, payload


class TestSmokeToken:
    def test_not_derivable_from_public_config_hash(self) -> None:
        """H2: the token must not be a plain hash of the (public) config_hash."""
        ch = "deadbeefcafe1234"
        plain = hashlib.sha256(f"audio_agent_smoke|{ch}".encode()).hexdigest()[:24]
        assert _safety.smoke_token(ch) != plain

    def test_roundtrip_verifies_and_rejects(self) -> None:
        ch = "abc123"
        tok = _safety.smoke_token(ch)
        assert _safety.verify_smoke_token(tok, ch) is True
        assert _safety.verify_smoke_token("wrong-token", ch) is False
        assert _safety.verify_smoke_token(tok, "other-hash") is False
        assert _safety.verify_smoke_token(None, ch) is False
        assert _safety.verify_smoke_token(tok, None) is False

    @pytest.mark.parametrize(
        ("label", "token"),
        [
            ("an en-dash from a chat UI", "9f3c1e–abcdef"),  # noqa: RUF001 - en dash is intentional in this message
            ("an accented character", "tokén-from-a-host"),
            ("a lone surrogate", "\ud800abc"),
            ("an int", 12345),
            ("a list", ["a"]),
            ("bytes", b"abc"),
        ],
    )
    def test_a_malformed_token_is_refused_rather_than_raised(self, label: str, token: object) -> None:
        """The token round-trips through the host LLM, so it arrives as whatever that produced.

        ``hmac.compare_digest`` refuses str arguments outside ASCII, so a token carrying one
        smart character -- what a chat UI does to a hex string in passing -- raised ``TypeError``
        straight out of ``run`` instead of refusing. Every verb's contract is to answer in JSON;
        a traceback tells the host nothing it can act on, and the actionable answer here is the
        ordinary one: the evidence does not match, go and smoke first.
        """
        assert _safety.verify_smoke_token(token, "abc123") is False, label

    def test_a_malformed_token_refuses_the_run_verb_in_json(self) -> None:
        """End-to-end, because the raise happened at the call site rather than in the helper."""
        from nemo_curator.audio_agent import verbs

        os.environ["AUDIO_AGENT_REQUIRE_SMOKE"] = "1"
        try:
            out = verbs.run(
                recipe={"stages": [{"ref": "MonoConversionStage", "params": {}}]},
                confirm=True,
                smoke_token="9f3c1e–abcdef",  # noqa: RUF001, S106 - en dash is intentional in this message
            )
        finally:
            os.environ.pop("AUDIO_AGENT_REQUIRE_SMOKE", None)

        assert out["status"] == "refused", out
        assert "smoke" in out["reason"], f"refused, but not over the token: {out['reason']}"
        json.dumps(out, default=str)  # the host has to be able to read the answer

    def test_secret_env_changes_token(self, monkeypatch) -> None:  # noqa: ANN001
        monkeypatch.setenv("AUDIO_AGENT_SMOKE_SECRET", "secret-a")
        _safety._smoke_secret.cache_clear()
        tok_a = _safety.smoke_token("h")
        monkeypatch.setenv("AUDIO_AGENT_SMOKE_SECRET", "secret-b")
        _safety._smoke_secret.cache_clear()
        tok_b = _safety.smoke_token("h")
        _safety._smoke_secret.cache_clear()
        assert tok_a != tok_b


class TestRequireSmoke:
    def test_env_toggle(self, monkeypatch) -> None:  # noqa: ANN001
        monkeypatch.delenv("AUDIO_AGENT_REQUIRE_SMOKE", raising=False)
        assert _safety.require_smoke() is False
        for truthy in ("1", "true", "YES", "on"):
            monkeypatch.setenv("AUDIO_AGENT_REQUIRE_SMOKE", truthy)
            assert _safety.require_smoke() is True
        monkeypatch.setenv("AUDIO_AGENT_REQUIRE_SMOKE", "0")
        assert _safety.require_smoke() is False


class TestDualPurposePathParams:
    """model_path/tokenizer_path hold either a local path or a hub id; only paths are locked."""

    @pytest.mark.parametrize(
        "hub_id",
        [
            "nvidia/diar_sortformer_4spk-v1",  # SpeakerSeparationStage's real default
            "nvidia/parakeet-tdt-0.6b-v2",
            "openai/whisper-large-v3",
            "bert-base-uncased",
        ],
    )
    def test_hub_ids_are_not_treated_as_workspace_paths(self, hub_id: str) -> None:
        assert _safety.names_local_path(hub_id) is False

    @pytest.mark.parametrize(
        "path",
        ["/models/asr.nemo", "./local/band.joblib", "~/models/tok.model", "../shared/m.onnx", "models/asr.nemo"],
    )
    def test_local_artifacts_are_treated_as_paths(self, path: str) -> None:
        assert _safety.names_local_path(path) is True

    def test_default_recipe_with_a_hub_id_is_not_refused(self, monkeypatch, tmp_path) -> None:  # noqa: ANN001
        """Locking model_path by name alone would refuse the agent's own default recipe."""
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
        recipe = Recipe.from_dict(
            {"stages": [{"ref": "SpeakerSeparationStage", "params": {"model_path": "nvidia/diar_sortformer_4spk-v1"}}]}
        )
        assert _safety.path_violations(_safety.recipe_path_params(recipe)) == []

    def test_a_local_model_path_outside_the_workspace_is_blocked(self, monkeypatch, tmp_path) -> None:  # noqa: ANN001
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
        recipe = Recipe.from_dict(
            {"stages": [{"ref": "SpeakerSeparationStage", "params": {"model_path": "/etc/evil.nemo"}}]}
        )
        assert len(_safety.path_violations(_safety.recipe_path_params(recipe))) == 1

    def test_a_nested_path_is_never_mistaken_for_a_hub_id(self) -> None:
        """No hub id has a directory tree in it, so a second slash settles the question even
        for a value that is relative and does not exist."""
        assert _safety.names_local_path("out/models/mymodel") is True

    def test_classification_does_not_depend_on_where_the_agent_was_launched(self, monkeypatch, tmp_path) -> None:  # noqa: ANN001
        """The old existence probe ran against the working directory, so merely running next
        to a directory named ``nvidia/`` reclassified the default model id as a local file and
        refused the agent's own default recipe -- the same command passing or failing depending
        on where it was typed. Existence is a question about the workspace, not the CWD.
        """
        (tmp_path / "nvidia").mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("AUDIO_AGENT_WORKSPACE", raising=False)

        assert _safety.names_local_path("nvidia/diar_sortformer_4spk-v1") is False


class TestSharedDependencyLocationsAreNotLocked:
    """The lock governs the dataset and the outputs, not shared dependencies.

    ``cache_dir`` is a model-download cache and ``config_path`` a packaged pipeline config --
    closer to site-packages than to user data. Locking them refused the natural
    ``~/.cache/huggingface`` on validate, smoke and run alike with no override, so a locked
    deployment could no longer share one cache between projects and re-downloaded
    multi-gigabyte checkpoints per recipe.
    """

    def test_a_shared_model_cache_outside_the_workspace_is_allowed(self, monkeypatch, tmp_path) -> None:  # noqa: ANN001
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
        recipe = Recipe.from_dict(
            {"stages": [{"ref": "InferenceSortformerStage", "params": {"cache_dir": "~/.cache/huggingface"}}]}
        )
        assert _safety.path_violations(_safety.recipe_path_params(recipe)) == []

    def test_a_packaged_config_outside_the_workspace_is_allowed(self, monkeypatch, tmp_path) -> None:  # noqa: ANN001
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
        recipe = Recipe.from_dict(
            {"stages": [{"ref": "AudioDataFilterStage", "params": {"config_path": "/opt/pkg/default_config.yaml"}}]}
        )
        assert _safety.path_violations(_safety.recipe_path_params(recipe)) == []

    def test_the_dataset_and_outputs_are_still_locked(self, monkeypatch, tmp_path) -> None:  # noqa: ANN001
        """The exemption is narrow: everything the run reads as data or writes as output stays
        contained, which is what the lock was opted into for."""
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
        recipe = Recipe.from_dict(
            {"stages": [{"ref": "ManifestWriterStage", "params": {"output_path": "/etc/out.jsonl"}}]}
        )
        assert len(_safety.path_violations(_safety.recipe_path_params(recipe))) == 1


class TestWorkspaceConfigValidation:
    """A misconfigured lock must never read as an absent one."""

    @pytest.mark.parametrize("bad", ["relative_ws", "/nonexistent/path/xyz"])
    def test_invalid_workspace_fails_closed(self, monkeypatch, bad: str) -> None:  # noqa: ANN001
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", bad)
        assert _safety.workspace_config_error() is not None
        assert _safety.workspace_root() is None
        # Any path at all: a misconfigured lock must reject rather than silently allow.
        assert _safety.path_violations([os.path.join(tempfile.gettempdir(), "anything")]) != []

    def test_valid_workspace_is_unaffected(self, monkeypatch, tmp_path) -> None:  # noqa: ANN001
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
        assert _safety.workspace_config_error() is None
        assert _safety.path_violations([str(tmp_path / "a.jsonl")]) == []


def test_an_empty_secret_file_heals_instead_of_stranding_smoke_evidence(tmp_path: Path) -> None:
    """A 0-byte secret used to be terminal, and silently so.

    ``os.link`` kept failing against the dead file, so every process fell back to its own
    per-process key: ``AUDIO_AGENT_REQUIRE_SMOKE`` then refused forever, which is exactly
    the cross-process failure the shared secret exists to prevent. Reachable from a crash
    mid-write, a full disk, or a stray ``touch``.
    """
    path = tmp_path / "audio_agent_smoke.secret"
    path.write_bytes(b"")

    first = _safety._read_or_create_secret(str(path))
    second = _safety._read_or_create_secret(str(path))

    assert first, "an unusable file must be replaced, not inherited"
    assert first == second, "every process must end up with the same shared secret"
    assert path.stat().st_mode & 0o777 == 0o600, "the secret stays owner-only"


def test_a_concurrent_creator_never_overwrites_the_winners_secret(tmp_path: Path) -> None:
    """Create-once-share-the-winner, delegated to utils/atomic_io."""
    path = tmp_path / "audio_agent_smoke.secret"

    winner = _safety._read_or_create_secret(str(path))
    again = _safety._read_or_create_secret(str(path))

    assert winner == again
    assert _safety._stored_secret(path) == winner


def test_the_secret_dir_expands_a_tilde_like_the_run_store_does(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One env var must not mean two different directories.

    ``run_store.runs_dir`` expands ``AUDIO_AGENT_RUNS_DIR``; the secret beside it did not.
    A tilde-valued setting therefore put the run records in the real home and the secret in
    a literal "~" folder under the working directory -- so the secret moved with the CWD and
    stopped being shared, which is the only thing it exists to do.
    """
    from nemo_curator.audio_agent import run_store

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", "~/agent_runs")

    first = _safety._secret_dir_candidates()[0]

    assert "~" not in first
    assert first.startswith(run_store.runs_dir()), "both resolve under the same home"


def test_no_home_still_never_proposes_a_tilde_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    """``expanduser`` returns the input unchanged with no passwd entry, so the guard has to
    catch a tilde that survived expansion -- not just a bare "~"."""
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", "~/agent_runs")

    assert [c for c in _safety._secret_dir_candidates() if "~" in c] == []
