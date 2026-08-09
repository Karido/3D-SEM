"""Integrity-checked download of the optional U-Net weights which
are stored as a release asset of version 1.0.0 on GitHub."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

_WEIGHTS_URL = "https://github.com/Karido/3D-SEM/releases/download/v1.0.0/sem3d-unet-weights.h5"
_WEIGHTS_SHA256 = "09358399d8be783c8f5b100aa4166b706946fa6a6adc8562c2bd6358282d46d4"
_WEIGHTS_NAME = "sem3d-unet-weights.h5"


def _cache_dir() -> Path:
    """Return (and create if needed) the local cache directory path for storing weights."""
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    weights_dir = root / "sem3d" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    return weights_dir


def _sha256(file_path: Path) -> str:
    """Compute the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_unet_weights() -> Path:
    """Return a local path to the verified weights, downloading if necessary."""
    target = _cache_dir() / _WEIGHTS_NAME
    if target.exists() and _sha256(target) == _WEIGHTS_SHA256:
        return target
    tmp = target.with_suffix(".part")
    urllib.request.urlretrieve(_WEIGHTS_URL, tmp)
    digest = _sha256(tmp)
    if digest != _WEIGHTS_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum Mismatch: Downloaded weights have SHA256 {digest}, expected {_WEIGHTS_SHA256}")
    tmp.replace(target)
    return target
