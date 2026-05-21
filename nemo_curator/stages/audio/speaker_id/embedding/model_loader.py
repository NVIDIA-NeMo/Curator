"""Load WeSpeaker models with device placement.

Loads local model directories containing ``config.yaml`` + ``avg_model.pt``
by loading model source files directly by file path via importlib.util,
completely bypassing ``wespeaker/__init__.py`` which unconditionally imports
frontends/transformers/torchvision and crashes in containers.
"""

import importlib.util
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import yaml

logger = logging.getLogger(__name__)

_MODEL_PREFIX_TO_MODULE = {
    "ResNet": "resnet",
    "XVEC": "tdnn",
    "ECAPA_TDNN": "ecapa_tdnn",
    "REPVGG": "repvgg",
    "CAMPPlus": "campplus",
    "ERes2Net": "eres2net",
    "Res2Net": "res2net",
    "Gemini": "gemini_dfresnet",
    "ReDimNet": "redimnet",
    "SimAM_ResNet": "samresnet",
    "XI_VEC": "xi_vector",
}


def _find_wespeaker_models_dir() -> str:
    """Locate the wespeaker/models/ directory on sys.path.

    When multiple installs exist (e.g. container + pip), picks the one
    whose resnet.py defines ``ResNet293`` (the version we need).
    Falls back to the last candidate found.
    """
    candidates = []
    for p in sys.path:
        candidate = os.path.join(p, "wespeaker", "models")
        if os.path.isdir(candidate):
            candidates.append(candidate)
    if not candidates:
        raise ImportError("Cannot find wespeaker package in sys.path")
    for c in candidates:
        resnet_py = os.path.join(c, "resnet.py")
        if os.path.isfile(resnet_py):
            with open(resnet_py, encoding="utf-8") as f:
                src = f.read()
            if "ResNet293" in src:
                return c
    return candidates[-1]


def _load_py_file(filepath: str, fake_name: str):
    """Load a .py file as a module using a fake name to avoid triggering parent __init__."""
    spec = importlib.util.spec_from_file_location(fake_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fake_name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class LoadedModel:
    """Container for a loaded WeSpeaker model + metadata."""

    model: torch.nn.Module
    frontend_type: str
    device: torch.device
    embedding_dim: int


def _ensure_wespeaker_stubs():
    """Insert stub packages for wespeaker and wespeaker.models into sys.modules.

    This prevents Python from executing wespeaker/__init__.py (which drags in
    frontends/transformers/torchvision) when model files do
    ``import wespeaker.models.pooling_layers``.
    """
    import types

    for pkg_name in ("wespeaker", "wespeaker.models"):
        if pkg_name not in sys.modules:
            stub = types.ModuleType(pkg_name)
            stub.__path__ = []
            stub.__package__ = pkg_name
            sys.modules[pkg_name] = stub


def _get_model_class(model_name: str):
    """Resolve a WeSpeaker model class by name, loading .py files by path."""
    models_dir = _find_wespeaker_models_dir()
    _ensure_wespeaker_stubs()

    pooling_key = "wespeaker.models.pooling_layers"
    if pooling_key not in sys.modules:
        pooling_path = os.path.join(models_dir, "pooling_layers.py")
        if os.path.isfile(pooling_path):
            _load_py_file(pooling_path, pooling_key)

    for prefix, module_file in _MODEL_PREFIX_TO_MODULE.items():
        if model_name.startswith(prefix):
            mod_key = f"wespeaker.models.{module_file}"
            if mod_key not in sys.modules:
                filepath = os.path.join(models_dir, f"{module_file}.py")
                if not os.path.isfile(filepath):
                    raise FileNotFoundError(f"Model file not found: {filepath}")
                _load_py_file(filepath, mod_key)
            return getattr(sys.modules[mod_key], model_name)

    raise ValueError(
        f"Unknown model '{model_name}'. "
        f"Supported prefixes: {list(_MODEL_PREFIX_TO_MODULE.keys())}"
    )


def _load_model_from_dir(model_dir: str, device: str) -> torch.nn.Module:
    """Load a WeSpeaker model from a local directory.

    Reads config.yaml for model class + args, instantiates the model,
    and loads avg_model.pt weights.
    """
    config_path = os.path.join(model_dir, "config.yaml")
    weight_path = os.path.join(model_dir, "avg_model.pt")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing {config_path}")
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Missing {weight_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.debug("Model config: %s", config)

    model_name = config["model"]
    model_args = config.get("model_args", {})

    model_cls = _get_model_class(model_name)
    model = model_cls(**model_args)

    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint, strict=False)

    return model


def load_wespeaker_model(
    model_name_or_path: str,
    device: str = "cuda:0",
    model_cache_dir: Optional[str] = None,
) -> LoadedModel:
    """Load a WeSpeaker model and move it to *device*.

    Args:
        model_name_or_path: Path to a local model directory containing
            ``config.yaml`` and ``avg_model.pt``.
        device: PyTorch device string.
        model_cache_dir: Unused, kept for API compatibility.

    Returns:
        LoadedModel with the model in eval mode on the requested device.
    """
    if not os.path.isdir(model_name_or_path):
        raise FileNotFoundError(
            f"Model directory not found: {model_name_or_path}. "
            f"Hub download is not supported; provide a local path."
        )

    logger.info("Loading WeSpeaker model: %s", model_name_or_path)
    nn_model = _load_model_from_dir(model_name_or_path, device)
    nn_model = nn_model.to(device)
    nn_model.eval()

    frontend_type = getattr(nn_model, "frontend_type", "fbank")

    emb_dim = _infer_embedding_dim(nn_model, device, frontend_type)
    logger.info(
        "Model loaded — frontend=%s  emb_dim=%d  device=%s",
        frontend_type, emb_dim, device,
    )
    return LoadedModel(
        model=nn_model,
        frontend_type=frontend_type,
        device=torch.device(device),
        embedding_dim=emb_dim,
    )


def _infer_embedding_dim(model: torch.nn.Module, device: str, frontend_type: str) -> int:
    """Run a tiny dummy forward pass to discover the embedding dimension."""
    try:
        dev = torch.device(device)
        if frontend_type == "fbank":
            dummy = torch.randn(1, 200, 80, device=dev)
        else:
            dummy = torch.randn(1, 200, 1024, device=dev)
        with torch.no_grad():
            out = model(dummy)
            emb = out[-1] if isinstance(out, tuple) else out
        return emb.shape[-1]
    except Exception:
        logger.warning("Could not infer embedding dim, defaulting to 256")
        return 256
