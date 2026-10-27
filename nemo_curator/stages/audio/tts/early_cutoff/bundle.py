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

"""The early cut-off endpoint bundle: the gate checkpoints and the MFA verifier calibration.

Both :class:`~.endpoint_gate.EarlyCutOffEndpointGateStage` and :class:`~.mfa_resolver.EarlyCutOffMFAResolverStage`
read the same files::

    english_endpoint.pt          multilingual_endpoint.pt       the gate's CTC endpoint heads
    english_verifier.json        multilingual_verifier.json     verifier calibration
    multilingual_thresholds.json mfa_language_models.yaml       per-language thresholds and MFA model map
    manifest.json                (optional) sha256 of each artifact, checked on load

Point ``bundle_dir`` at a local copy with these names. Otherwise the checkpoints are downloaded from ``bundle_url``
(a folder URL) into ``~/.cache/nemo_curator/granary_hifi/early_cutoff/`` on first use, and the calibration files
shipped in ``calibration/`` next to this module are used.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_BUNDLE_URL: str | None = (
    "https://huggingface.co/datasets/nvidia/Granary-HiFi/resolve/main/assets/quality_control/"
)
CALIBRATION_DIR = Path(__file__).parent / "calibration"

ENGLISH_CHECKPOINT = "english_endpoint.pt"
ENGLISH_CALIBRATION = "english_verifier.json"
MULTILINGUAL_CHECKPOINT = "multilingual_endpoint.pt"
MULTILINGUAL_CALIBRATION = "multilingual_verifier.json"
MULTILINGUAL_THRESHOLDS = "multilingual_thresholds.json"
MFA_LANGUAGE_MODELS = "mfa_language_models.yaml"
BUNDLE_MANIFEST = "manifest.json"

# bundle name -> (file name under ``bundle_url``, sha256)
CHECKPOINT_FILES: dict[str, tuple[str, str]] = {
    ENGLISH_CHECKPOINT: (
        "early_cutoff_english.pt",
        "a3c2bcc96f8566f6874de8315e2d559162787f6f418120622363989a9f0a6962",  # pragma: allowlist secret
    ),
    MULTILINGUAL_CHECKPOINT: (
        "early_cutoff_multilingual.pt",
        "e25f3ea43afcde35f69861068aaaff5b1b7a4d83aba7a47927a953b8ea5f66b7",  # pragma: allowlist secret
    ),
}

# bundle name -> file name in ``CALIBRATION_DIR``
CALIBRATION_FILES: dict[str, str] = {
    ENGLISH_CALIBRATION: "early_cutoff_english_verifier.json",
    MULTILINGUAL_CALIBRATION: "early_cutoff_multilingual_verifier.json",
    MULTILINGUAL_THRESHOLDS: "early_cutoff_multilingual_thresholds.json",
    MFA_LANGUAGE_MODELS: "early_cutoff_mfa_language.yaml",
}


def resolve_bundle(bundle_dir: str | None, bundle_url: str | None) -> dict[str, Path] | None:
    """Bundle name -> local path: every file under ``bundle_dir`` if set, else downloaded checkpoints + shipped calibration.

    Returns ``None`` when neither is configured; the stages then leave their candidates at ``REVIEW``.
    """
    if bundle_dir:
        root = Path(bundle_dir).expanduser()
        names = (*CHECKPOINT_FILES, *CALIBRATION_FILES, BUNDLE_MANIFEST)
        return {name: root / name for name in names}
    if not bundle_url:
        return None
    from nemo_curator.stages.audio.tts.model_download import cache_dir, download_file

    root = cache_dir("early_cutoff")
    base = bundle_url if bundle_url.endswith("/") else bundle_url + "/"
    files = {
        name: download_file(base + remote_name, root / name, sha256=sha256)
        for name, (remote_name, sha256) in CHECKPOINT_FILES.items()
    }
    files.update({name: CALIBRATION_DIR / shipped for name, shipped in CALIBRATION_FILES.items()})
    return files
