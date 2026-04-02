import base64
import configparser
import io
import mimetypes
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch

# Global S3 client (initialized lazily)
_s3_client = None


def _is_valid_value(value: str | None) -> bool:
    """Return False if value is missing, NaN, or empty string."""
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    return not (isinstance(value, str) and not value.strip())


def _is_remote_storage_path(value: str) -> bool:
    """Return True if value is an S3 or AIS object storage path."""
    if not value or not isinstance(value, str):
        return False
    s = value.strip()
    return s.startswith(("s3://", "ais://"))


def _parse_s3cfg(config_path: str = "~/.s3cfg", section: str = "default") -> dict:
    """Parse an .s3cfg file and return credentials."""
    path = Path(config_path).expanduser()
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)
    config = configparser.ConfigParser()
    config.read(path)
    if section not in config:
        msg = f"No [{section}] section found in {path}"
        raise ValueError(msg)
    return {
        "use_https": config.getboolean(section, "use_https", fallback=True),
        "access_key": config.get(section, "access_key", fallback=None),
        "secret_key": config.get(section, "secret_key", fallback=None),
        "bucket_location": config.get(section, "bucket_location", fallback=None),
        "host_base": config.get(section, "host_base", fallback=None),
        "authn_token": config.get(section, "authn_token", fallback=None),
    }


class _AISClient:
    """Thin S3-compatible client for AIStore using Bearer token auth."""

    def __init__(self, endpoint_url: str, token: str):
        import requests as _requests

        self._base = endpoint_url.rstrip("/")
        self._session = _requests.Session()
        self._session.headers["Authorization"] = f"Bearer {token}"

    def get_object(self, Bucket: str, Key: str, **_kwargs: str) -> dict:  # noqa: N803
        """Fetch an object from AIStore via S3-compatible endpoint."""
        from botocore.exceptions import ClientError

        _http_error_threshold = 400
        url = f"{self._base}/s3/{Bucket}/{Key}"
        resp = self._session.get(url)
        if resp.status_code >= _http_error_threshold:
            raise ClientError(
                {"Error": {"Code": str(resp.status_code), "Message": resp.reason}},
                "GetObject",
            )
        return {"Body": io.BytesIO(resp.content)}


def _init_s3_client(s3cfg: str) -> None:
    """Initialize the global S3 client from an s3cfg path like ``~/.s3cfg[default]``."""
    global _s3_client  # noqa: PLW0603
    if _s3_client is not None:
        return
    path, section = s3cfg.rsplit("[", 1)
    cfg = _parse_s3cfg(path, section.rstrip("]"))
    endpoint_url = ("https://" if cfg["use_https"] else "http://") + cfg["host_base"]
    if cfg.get("authn_token"):
        _s3_client = _AISClient(endpoint_url, cfg["authn_token"])
    else:
        import boto3
        from botocore.config import Config

        _s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=cfg["access_key"],
            aws_secret_access_key=cfg["secret_key"],
            region_name=cfg["bucket_location"],
            config=Config(connect_timeout=5),
        )


def _parse_s3_path(s3_path: str) -> tuple[str, str]:
    """Parse ``s3://bucket/key`` into ``(bucket, key)``."""
    parsed = urlparse(str(s3_path))
    return parsed.netloc, parsed.path.lstrip("/")


def _read_remote_bytes(path: str) -> bytes:
    """Read bytes from a remote storage path using the global S3 client."""
    if _s3_client is None:
        msg = "S3 client not initialized. Set s3cfg on PrepareOmniRequestStage."
        raise RuntimeError(msg)
    bucket, key = _parse_s3_path(path)
    response = _s3_client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def _is_local_path(value: str) -> bool:
    """Return True if value looks like a local file path (not http/https)."""
    if not value or not isinstance(value, str):
        return False
    s = value.strip()
    return not s.startswith(("http://", "https://", "s3://", "ais://"))


def _parse_dali_webdataset_index(index_path: Path) -> dict[str, tuple[int, int]]:
    """Parse a DALI ``wds2idx.py`` v1.2 index into ``member_name -> (data_offset, size)``.

    Each non-header line lists components as
    ``ext offset size name`` (repeated per component in the sample), matching
    `NVIDIA DALI tools/wds2idx.py <https://github.com/NVIDIA/DALI/blob/main/tools/wds2idx.py>`_.
    """
    raw = index_path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        msg = "DALI index file is empty: " + str(index_path)
        raise ValueError(msg)
    header_parts = lines[0].split()
    min_header_parts = 2
    if len(header_parts) < min_header_parts or not header_parts[0].startswith("v"):
        msg = "Invalid DALI webdataset index header: " + lines[0]
        raise ValueError(msg)
    out: dict[str, tuple[int, int]] = {}
    for line in lines[1:]:
        parts = line.split()
        n = len(parts)
        if n % 4 != 0:
            msg = (
                "Unparseable DALI index line (expected a multiple of 4 tokens "
                "ext/offset/size/name per component): " + line[:200]
            )
            raise ValueError(msg)
        for i in range(0, n, 4):
            _ext, off_s, sz_s, name = parts[i], parts[i + 1], parts[i + 2], parts[i + 3]
            out[name] = (int(off_s), int(sz_s))
    if not out:
        msg = "DALI index contains no member entries: " + str(index_path)
        raise ValueError(msg)
    return out


@dataclass
class PrepareOmniRequestStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    A stage to read a jsonl file where each line is a dictionary with the following keys:
    {"system_text": "You are a helpful assistant.", "text": "Say hello to the user.", "image_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"}
    Making LLM messages with following format:
    [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"}},
        {"type": "audio_url", "audio_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"}},
        {"type": "text", "text": "What can you see and hear? Answer in one sentence."}
    ]}
    ]
    and return a DocumentBatch with the LLM messages.

    Args:
        format: How to encode audio content parts.
            ``"data_url"`` (default) — ``audio_url`` with ``data:`` URI (vLLM).
            ``"input_data"`` — ``input_audio`` with raw base64 + format (OpenAI / NIM).
            Images always use ``image_url`` regardless of this setting.
        input_tar: Path to a tar archive. When set, ``audio_filepath_key`` /
            ``image_filepath_key`` values are **member names** inside the tar
            (as in ``tar tf``). With ``input_index``, bytes are read via the DALI
            index (fast random access). With an empty ``input_index``, members are
            resolved and read with the standard library ``tarfile`` module.
        input_index: Optional path to a DALI v1.2 index for ``input_tar``
            (``tools/wds2idx.py`` in the DALI repo). If empty, ``input_tar`` alone
            is enough. If set, ``input_tar`` must also be set.
    """

    name: str = "MakeMessagesStage"
    format: str = "data_url"
    audio_filepath_key: str = "audio_filepath"
    image_filepath_key: str = "image_url"
    input_tar: str = ""
    input_index: str = ""
    user_prompt: str = "text"
    system_prompt: str = "system_text"
    s3cfg: str = ""

    _tar_member_index: dict[str, tuple[int, int]] | None = field(default=None, init=False, repr=False)
    _tar_member_names_cache: list[str] | None = field(default=None, init=False, repr=False)

    def _uses_input_tar(self) -> bool:
        return bool(self.input_tar.strip())

    def _uses_dali_index(self) -> bool:
        return bool(self.input_tar.strip()) and bool(self.input_index.strip())

    def _ensure_index_has_tar(self) -> None:
        if self.input_index.strip() and not self.input_tar.strip():
            msg = "input_index requires input_tar to be set."
            raise ValueError(msg)

    def _tar_member_index_map(self) -> dict[str, tuple[int, int]]:
        if self._tar_member_index is None:
            idx_path = Path(self.input_index.strip()).expanduser().resolve()
            self._tar_member_index = _parse_dali_webdataset_index(idx_path)
        return self._tar_member_index

    def _tar_member_names_list(self) -> list[str]:
        if self._tar_member_names_cache is None:
            tar_path = Path(self.input_tar.strip()).expanduser().resolve()
            if not tar_path.is_file():
                msg = "input_tar is not a file: " + str(tar_path)
                raise FileNotFoundError(msg)
            with tarfile.open(tar_path, "r:*") as tf:
                self._tar_member_names_cache = [m.name for m in tf.getmembers() if m.isfile()]
        return self._tar_member_names_cache

    def _resolve_tar_member_name(self, key: str) -> str:
        """Map a row value to a tar member name (DALI index or ``tarfile`` listing)."""
        key = key.strip()
        if self._uses_dali_index():
            m = self._tar_member_index_map()
            if key in m:
                return key
            basename = Path(key).name
            matches = [n for n in m if n == key or n.endswith("/" + key) or Path(n).name == basename]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                msg = f"Ambiguous tar member key {key!r}; matches include {matches[:8]!r}"
                raise ValueError(msg)
            msg = f"Member not found in DALI index for key {key!r}"
            raise FileNotFoundError(msg)

        names = self._tar_member_names_list()
        if key in names:
            return key
        basename = Path(key).name
        matches = [n for n in names if n == key or n.endswith("/" + key) or Path(n).name == basename]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            msg = f"Ambiguous tar member key {key!r}; matches include {matches[:8]!r}"
            raise ValueError(msg)
        msg = f"Member not found in tar for key {key!r}"
        raise FileNotFoundError(msg)

    def _read_bytes_from_tar(self, member_key: str) -> tuple[bytes, str]:
        """Read raw file bytes from ``input_tar``; optional DALI index for seek+read."""
        self._ensure_index_has_tar()
        tar_path = Path(self.input_tar.strip()).expanduser().resolve()
        if not tar_path.is_file():
            msg = "input_tar is not a file: " + str(tar_path)
            raise FileNotFoundError(msg)
        name = self._resolve_tar_member_name(member_key)

        if self._uses_dali_index():
            offset, size = self._tar_member_index_map()[name]
            with tar_path.open("rb") as f:
                f.seek(offset)
                raw = f.read(size)
            if len(raw) != size:
                msg = f"Short read from tar for {name!r}: got {len(raw)} bytes, expected {size}"
                raise OSError(msg)
            return raw, name

        with tarfile.open(tar_path, "r:*") as tf:
            try:
                info = tf.getmember(name)
            except KeyError as exc:
                msg = f"Member not found in tar: {name!r}"
                raise FileNotFoundError(msg) from exc
            if not info.isfile():
                msg = f"Tar entry is not a regular file: {name!r}"
                raise OSError(msg)
            stream = tf.extractfile(info)
            if stream is None:
                msg = f"Cannot read tar member payload: {name!r}"
                raise OSError(msg)
            raw = stream.read()
        return raw, name

    def _local_file_to_data_url(self, path: str) -> tuple[str, str]:
        """Read a local file, encode to base64, return a data URL."""
        if self._uses_input_tar():
            raw, resolved_name = self._read_bytes_from_tar(path)
            mime_type, _ = mimetypes.guess_type(resolved_name)
            if mime_type is None:
                mime_type = "application/octet-stream"
            b64 = base64.standard_b64encode(raw).decode("ascii")
            return b64, mime_type

        if _is_remote_storage_path(path):
            raw = _read_remote_bytes(path)
            mime_type, _ = mimetypes.guess_type(path)
            if mime_type is None:
                mime_type = "application/octet-stream"
            b64 = base64.standard_b64encode(raw).decode("ascii")
            return b64, mime_type

        path_obj = Path(path).expanduser().resolve()
        if not path_obj.is_file():
            msg = "Local path is not a file: " + path
            raise FileNotFoundError(msg)
        mime_type, _ = mimetypes.guess_type(str(path_obj))
        if mime_type is None:
            mime_type = "application/octet-stream"
        raw = path_obj.read_bytes()
        b64 = base64.standard_b64encode(raw).decode("ascii")
        return b64, mime_type

    def _file_to_base64(self, path: str) -> tuple[str, str]:
        """Read a file (audio or image), return ``(base64_data, extension)``.

        Used for ``input_audio`` content parts where the API expects raw base64
        bytes plus a short format string (e.g. ``"wav"``, ``"png"``).
        Supports tar, S3/AIS remote storage, and local files.
        """
        if self._uses_input_tar():
            raw, resolved_name = self._read_bytes_from_tar(path)
            ext = Path(resolved_name).suffix.lstrip(".").lower()
            if not ext:
                ext = "bin"
            b64 = base64.standard_b64encode(raw).decode("ascii")
            return b64, ext

        if _is_remote_storage_path(path):
            raw = _read_remote_bytes(path)
            ext = Path(path).suffix.lstrip(".").lower()
            if not ext:
                ext = "bin"
            b64 = base64.standard_b64encode(raw).decode("ascii")
            return b64, ext

        path_obj = Path(path).expanduser().resolve()
        if not path_obj.is_file():
            msg = "Local path is not a file: " + path
            raise FileNotFoundError(msg)
        raw = path_obj.read_bytes()
        b64 = base64.standard_b64encode(raw).decode("ascii")
        ext = path_obj.suffix.lstrip(".").lower()
        if not ext:
            ext = "bin"
        return b64, ext

    def _url_or_data_url(self, value: str, content_format: str) -> dict:
        """Encode a media reference as an OpenAI-compatible content part.

        Args:
            value: File path (local, tar member, S3/AIS) or HTTP(S) URL.
            content_format: Content part type to produce:
                ``"audio_url"``   — ``data:`` URL (vLLM).
                ``"input_audio"`` — raw base64 + format (OpenAI Chat / NIM).
                ``"image_url"``   — ``data:`` URL (universal).
        """
        if not _is_local_path(value) and not _is_remote_storage_path(value):
            return {"type": content_format, content_format: {"url": value}}

        if content_format == "input_audio":
            b64, ext = self._file_to_base64(value)
            return {"type": content_format, content_format: {"data": b64, "format": ext}}
        if content_format in ("audio_url", "image_url"):
            b64, mime_type = self._local_file_to_data_url(value)
            return {"type": content_format, content_format: {"url": f"data:{mime_type};base64,{b64}"}}
        msg = f"Invalid content_format: {content_format!r}. Supported: input_audio, audio_url, image_url."
        raise ValueError(msg)

    def _user_content_format_for_media(self, *, image: bool) -> str:
        """Map stage ``format`` to API content part type.

        Images always use ``image_url`` (accepted by both vLLM and OpenAI).
        Audio uses ``audio_url`` (vLLM) or ``input_audio`` (OpenAI / NIM)
        depending on the ``format`` setting.
        """
        if image:
            return "image_url"
        if self.format == "data_url":
            return "audio_url"
        if self.format == "input_data":
            return "input_audio"
        msg = f"Invalid format: {self.format!r}. Supported: 'data_url', 'input_data'."
        raise ValueError(msg)

    def _row_to_messages(self, row: dict) -> list[dict]:
        """Build LLM messages from a row with optional system_text, text, image_url, audio_url.

        Args:
            row: A dictionary with optional keys: system_text, text, image_url, audio_url.
        """
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        content_parts: list[dict] = []
        if self.image_filepath_key in row and _is_valid_value(row.get(self.image_filepath_key)):
            content_fmt = self._user_content_format_for_media(image=True)
            msg = self._url_or_data_url(row[self.image_filepath_key], content_format=content_fmt)
            content_parts.append(msg)

        if self.audio_filepath_key in row and _is_valid_value(row.get(self.audio_filepath_key)):
            content_fmt = self._user_content_format_for_media(image=False)
            msg = self._url_or_data_url(row[self.audio_filepath_key], content_format=content_fmt)
            content_parts.append(msg)

        text_val = self.user_prompt
        if text_val is not None and not (isinstance(text_val, float) and pd.isna(text_val)) and str(text_val).strip():
            content_parts.append({"type": "text", "text": text_val})

        if content_parts:
            messages.append({"role": "user", "content": content_parts})
        elif not messages:
            messages.append({"role": "user", "content": ""})

        return messages

    def process(self, input_batch: DocumentBatch) -> DocumentBatch:
        """Build LLM messages from each row and add a 'messages' column."""
        if self.s3cfg:
            _init_s3_client(self.s3cfg)
        df = input_batch.to_pandas().copy()
        df["messages"] = df.apply(lambda row: self._row_to_messages(row.to_dict()), axis=1)
        return DocumentBatch(
            data=df,
            dataset_name=input_batch.dataset_name,
            task_id=input_batch.task_id,
        )
