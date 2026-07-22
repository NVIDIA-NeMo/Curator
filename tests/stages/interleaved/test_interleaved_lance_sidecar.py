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

import json
import sqlite3
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pytest

from nemo_curator.stages import interleaved
from nemo_curator.stages.interleaved.lance import (
    LanceRowIdImageMaterializationStage,
    LanceTableConfig,
    ShardedSqliteUrlLanceAddressResolutionStage,
)
from nemo_curator.stages.interleaved.lance.sidecar import (
    build_sharded_sqlite_url_lance_sidecar,
    decode_rowaddr,
    decode_uint64,
    encode_uint64,
    hash_url,
    read_lance_url_sidecar_manifest,
    shard_for_digest,
)
from nemo_curator.tasks import InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA

if TYPE_CHECKING:
    from pathlib import Path


def _write_sharded_sidecar(
    path: Path,
    *,
    rows: list[tuple[str, int, int | None]],
    schema_version: int,
    shard_count: int = 4,
) -> None:
    shards = path / "shards"
    shards.mkdir(parents=True)
    for shard_id in range(shard_count):
        conn = sqlite3.connect(shards / f"shard-{shard_id:05d}.sqlite")
        if schema_version >= 2:
            conn.execute(
                "CREATE TABLE kv (url_hash BLOB PRIMARY KEY, row_id BLOB NOT NULL, rowaddr BLOB NOT NULL) WITHOUT ROWID"
            )
        else:
            conn.execute("CREATE TABLE kv (url_hash BLOB PRIMARY KEY, row_id BLOB NOT NULL) WITHOUT ROWID")
        conn.close()

    for url, row_id, rowaddr in rows:
        digest = hash_url(url)
        shard_id = shard_for_digest(digest, shard_count)
        conn = sqlite3.connect(shards / f"shard-{shard_id:05d}.sqlite")
        if schema_version >= 2:
            assert rowaddr is not None
            conn.execute("INSERT INTO kv VALUES (?, ?, ?)", (digest, encode_uint64(row_id), encode_uint64(rowaddr)))
        else:
            conn.execute("INSERT INTO kv VALUES (?, ?)", (digest, encode_uint64(row_id)))
        conn.commit()
        conn.close()

    manifest: dict[str, Any] = {
        "format": "sqlite-sharded-compact",
        "hash": "blake2b-128",
        "row_id_encoding": "uint64-little-endian",
        "sidecar_schema_version": schema_version,
        "shard_count": shard_count,
        "shards_dir": "shards",
    }
    if schema_version >= 2:
        manifest["rowaddr_encoding"] = "uint64-little-endian"
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _image_row(
    source_ref: str | None,
    *,
    binary_content: bytes | None = None,
    content_type: str | None = None,
) -> dict[str, Any]:
    return {
        "sample_id": "s1",
        "position": 0,
        "modality": "image",
        "content_type": content_type,
        "text_content": None,
        "binary_content": binary_content,
        "source_ref": source_ref,
        "materialize_error": None,
    }


def _task_for_urls(urls: list[str | None]) -> InterleavedBatch:
    rows = [_image_row(url) for url in urls]
    return InterleavedBatch(dataset_name="docs", data=pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA))


def test_lance_url_sidecar_hash_and_uint64_helpers() -> None:
    digest = hash_url("https://example.test/a.jpg")
    rowaddr = (7 << 32) | 42

    assert len(digest) == 16
    assert 0 <= shard_for_digest(digest, 8) < 8
    assert decode_uint64(encode_uint64(123)) == 123
    assert decode_rowaddr(rowaddr) == (7, 42)
    with pytest.raises(ValueError, match="expected 8 bytes"):
        decode_uint64(b"short")


def test_sharded_sqlite_url_lance_address_resolver_v1_row_id_only(tmp_path: Path) -> None:
    url = "https://example.test/a.jpg"
    _write_sharded_sidecar(tmp_path, rows=[(url, 123, None)], schema_version=1)

    stage = ShardedSqliteUrlLanceAddressResolutionStage(
        sidecar_dir=str(tmp_path),
        output_fragment_id_column=None,
        output_row_offset_column=None,
    )
    result = stage.process(_task_for_urls([url, "https://example.test/missing.jpg", "", None]))
    table = result.to_pyarrow()

    assert "lance_fragment_id" not in table.column_names
    assert "lance_row_offset" not in table.column_names
    assert table["lance_row_id"].combine_chunks().to_pylist() == [123, None, None, None]
    assert table["lance_address_present"].combine_chunks().to_pylist() == [True, False, False, False]
    assert table["lance_lookup_error"].combine_chunks().to_pylist() == [
        None,
        "not_found_in_sidecar",
        "missing_url",
        "missing_url",
    ]


def test_sharded_sqlite_url_lance_address_resolver_requires_rowaddr_for_default_outputs(tmp_path: Path) -> None:
    url = "https://example.test/a.jpg"
    _write_sharded_sidecar(tmp_path, rows=[(url, 123, None)], schema_version=1)

    stage = ShardedSqliteUrlLanceAddressResolutionStage(sidecar_dir=str(tmp_path))

    with pytest.raises(ValueError, match="does not contain Lance row addresses"):
        stage.process(_task_for_urls([url]))


def test_sharded_sqlite_url_lance_address_resolver_emits_split_row_address(tmp_path: Path) -> None:
    url = "https://example.test/a.jpg"
    rowaddr = (7 << 32) | 42
    _write_sharded_sidecar(tmp_path, rows=[(url, 123, rowaddr)], schema_version=2)

    stage = ShardedSqliteUrlLanceAddressResolutionStage(sidecar_dir=str(tmp_path))
    result = stage.process(_task_for_urls([url, "https://example.test/missing.jpg"]))
    table = result.to_pyarrow()

    assert table["lance_row_id"].combine_chunks().to_pylist() == [123, None]
    assert table["lance_fragment_id"].combine_chunks().to_pylist() == [7, None]
    assert table["lance_row_offset"].combine_chunks().to_pylist() == [42, None]
    assert table.schema.field("lance_fragment_id").type == pa.uint32()
    assert table.schema.field("lance_row_offset").type == pa.uint32()
    assert table["lance_address_present"].combine_chunks().to_pylist() == [True, False]
    assert table["lance_lookup_error"].combine_chunks().to_pylist() == [None, "not_found_in_sidecar"]


def test_sharded_sqlite_url_lance_address_resolver_is_lazy_exported() -> None:
    assert interleaved.ShardedSqliteUrlLanceAddressResolutionStage is ShardedSqliteUrlLanceAddressResolutionStage


def test_sharded_sqlite_url_lance_address_resolver_validates_config_and_input(tmp_path: Path) -> None:
    _write_sharded_sidecar(tmp_path, rows=[], schema_version=2)
    with pytest.raises(ValueError, match="configured together"):
        ShardedSqliteUrlLanceAddressResolutionStage(
            sidecar_dir=str(tmp_path),
            output_fragment_id_column="lance_fragment_id",
            output_row_offset_column=None,
        )
    with pytest.raises(ValueError, match="distinct"):
        ShardedSqliteUrlLanceAddressResolutionStage(
            sidecar_dir=str(tmp_path),
            output_row_id_column="duplicate",
            presence_column="duplicate",
        )

    stage = ShardedSqliteUrlLanceAddressResolutionStage(sidecar_dir=str(tmp_path), input_url_column="missing")
    with pytest.raises(ValueError, match="Input URL column"):
        stage.process(_task_for_urls(["https://example.test/a.jpg"]))


def test_build_url_lance_sidecar_and_materialize_with_split_row_address(tmp_path: Path) -> None:
    lance = pytest.importorskip("lance")

    dataset_path = tmp_path / "images.lance"
    table = pa.table(
        {
            "url": ["https://example.test/a.jpg", "https://example.test/b.jpg"],
            "image": [b"jpeg-a", b"jpeg-b"],
            "mime_type": ["image/jpeg", "image/jpeg"],
        },
        schema=pa.schema(
            [
                pa.field("url", pa.string()),
                pa.field("image", pa.large_binary()),
                pa.field("mime_type", pa.string()),
            ]
        ),
    )
    lance.write_dataset(
        table,
        str(dataset_path),
        mode="create",
        max_rows_per_file=1,
        max_rows_per_group=1,
        enable_stable_row_ids=True,
    )
    dataset = lance.dataset(str(dataset_path))
    sidecar_dir = tmp_path / "sidecar"

    report = build_sharded_sqlite_url_lance_sidecar(
        dataset=LanceTableConfig(uri=str(dataset_path), version=dataset.version),
        output_dir=sidecar_dir,
        shard_count=4,
        batch_size=1,
        insert_batch_rows=1,
        commit_every_rows=1,
        progress_every_rows=1,
        sample_url_count=2,
    )

    assert report["inserted_rows"] == 2
    manifest = read_lance_url_sidecar_manifest(sidecar_dir)
    assert manifest["row_count"] == 2
    assert (sidecar_dir / manifest["sample_urls"]).exists()

    resolver = ShardedSqliteUrlLanceAddressResolutionStage(sidecar_dir=str(sidecar_dir))
    resolved = resolver.process(_task_for_urls(["https://example.test/b.jpg"]))
    materializer = LanceRowIdImageMaterializationStage(
        dataset=LanceTableConfig(uri=str(dataset_path), version=dataset.version),
        address_mode="row_address",
        presence_column="lance_materialized_present",
        fetch_batch_size=1,
        io_threads=1,
    )
    try:
        materialized = materializer.process(resolved)
    finally:
        materializer.teardown()
        resolver.teardown()

    result = materialized.to_pyarrow()
    assert result["binary_content"].combine_chunks().to_pylist() == [b"jpeg-b"]
    assert result["content_type"].combine_chunks().to_pylist() == ["image/jpeg"]
    assert result["lance_materialized_present"].combine_chunks().to_pylist() == [True]


def test_build_url_lance_sidecar_cli_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tutorials.interleaved.lance import build_url_lance_sidecar as cli

    assert cli.parse_storage_options(["aws_region=us-east-1", "aws_endpoint=https://example.test"]) == {
        "aws_region": "us-east-1",
        "aws_endpoint": "https://example.test",
    }
    with pytest.raises(ValueError, match="KEY=VALUE"):
        cli.parse_storage_options(["missing-equals"])

    aws_dir = tmp_path / ".aws"
    aws_dir.mkdir()
    (aws_dir / "credentials").write_text(
        "[con]\naws_access_key_id=key\naws_secret_access_key=secret\naws_session_token=token\n",
        encoding="utf-8",
    )
    (aws_dir / "config").write_text(
        "[profile con]\nendpoint_url=https://pdx.s8k.io\nregion=us-east-1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    options = cli.aws_profile_storage_options("con")

    assert options["aws_access_key_id"] == "key"
    assert options["aws_secret_access_key"] == "secret"  # noqa: S105
    assert options["aws_session_token"] == "token"  # noqa: S105
    assert options["aws_endpoint"] == "https://pdx.s8k.io"
    assert options["aws_region"] == "us-east-1"
    assert options["aws_virtual_hosted_style_request"] == "false"


def test_build_url_lance_sidecar_cli_main_invokes_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tutorials.interleaved.lance import build_url_lance_sidecar as cli

    captured: dict[str, object] = {}

    def fake_build_sharded_sqlite_url_lance_sidecar(**kwargs: object) -> dict[str, int]:
        captured.update(kwargs)
        return {"inserted_rows": 3}

    monkeypatch.setattr(cli, "build_sharded_sqlite_url_lance_sidecar", fake_build_sharded_sqlite_url_lance_sidecar)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_url_lance_sidecar.py",
            "--image-uri",
            "memory://images",
            "--image-version",
            "4",
            "--output-dir",
            str(tmp_path / "sidecar"),
            "--key-column",
            "source_url",
            "--shard-count",
            "8",
            "--max-rows",
            "10",
            "--storage-option",
            "aws_region=us-east-1",
            "--overwrite",
        ],
    )

    cli.main()

    report = json.loads(capsys.readouterr().out)
    assert report == {"inserted_rows": 3}
    assert captured["dataset"] == LanceTableConfig(
        uri="memory://images",
        version=4,
        storage_options={"aws_region": "us-east-1"},
    )
    assert captured["output_dir"] == str(tmp_path / "sidecar")
    assert captured["key_column"] == "source_url"
    assert captured["shard_count"] == 8
    assert captured["max_rows"] == 10
    assert captured["overwrite"] is True
