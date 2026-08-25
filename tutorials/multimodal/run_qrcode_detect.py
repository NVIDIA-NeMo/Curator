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

"""Print QR code area ratio for images using the same logic as InterleavedQRCodeFilterStage."""

import argparse
from pathlib import Path
from urllib.request import urlopen

from nemo_curator.stages.interleaved.filter.qrcode_filter import _qr_code_ratio
from nemo_curator.stages.interleaved.utils import image_bytes_to_array


def _load_bytes(source: str) -> bytes:
    s = source.strip()
    if s.lower().startswith(("http://", "https://")):
        with urlopen(s) as resp:  # noqa: S310
            return resp.read()
    path = Path(s)
    if not path.is_file():
        msg = f"Not a file: {path}"
        raise SystemExit(msg)
    return path.read_bytes()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Detect QR codes via OpenCV (same as nemo_curator interleaved qrcode_filter): "
            "prints fraction of image area covered by QR polygon(s)."
        )
    )
    parser.add_argument(
        "sources",
        type=str,
        nargs="+",
        help="Image file path(s) and/or http(s) URL(s) (JPEG/PNG).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Same as InterleavedQRCodeFilterStage.score_threshold; images with ratio >= this are filtered out.",
    )
    args = parser.parse_args()

    for src in args.sources:
        raw = _load_bytes(src)
        arr = image_bytes_to_array(raw)
        ratio = _qr_code_ratio(arr)
        would_keep = ratio < args.threshold
        print(f"{src}")
        print(f"  qr_area_ratio={ratio:.6f}  threshold={args.threshold}  keep={would_keep}")


if __name__ == "__main__":
    main()
