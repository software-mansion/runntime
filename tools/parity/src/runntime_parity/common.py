"""What every reference script shares: where references go, which devices
the reference runs on, and the files a run saves."""

import json
import os
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from safetensors.numpy import save_file


def references_dir(family: str) -> Path:
    """The folder the parity tests read `family` from: `RUNNTIME_PARITY_DIR`,
    or `~/.cache/runntime-parity`."""
    root = os.environ.get("RUNNTIME_PARITY_DIR") or Path.home() / ".cache" / "runntime-parity"
    path = Path(root) / family
    path.mkdir(parents=True, exist_ok=True)
    return path


def devices() -> tuple[str, str]:
    """The device the reference outputs come from, and a second one whose
    outputs give the reference noise."""
    if torch.cuda.is_available():
        return "cpu", "cuda"
    if torch.backends.mps.is_available():
        return "cpu", "mps"
    raise RuntimeError("the reference noise needs a second device, CUDA or MPS")


def noise(first: NDArray[np.float32], second: NDArray[np.float32]) -> float:
    """The largest difference between two runs of the reference."""
    return float(np.abs(first - second).max())


def cosine_noise(first: NDArray[np.float32], second: NDArray[np.float32]) -> float:
    """The largest cosine distance between the rows of two runs of the
    reference."""
    a, b = first.astype(np.float64), second.astype(np.float64)
    cosine = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return float((1 - cosine).max())


def save(
    family: str,
    inputs: object,
    tensors: dict[str, NDArray[np.float32] | NDArray[np.int32]],
    meta: dict[str, object],
    libraries: list[str],
) -> None:
    """Writes `inputs.json`, `reference.safetensors` and `meta.json`, with the
    version of Python and of each library in `libraries`."""
    out = references_dir(family)
    (out / "inputs.json").write_text(json.dumps(inputs, ensure_ascii=False, indent=2))
    save_file(tensors, out / "reference.safetensors")
    versions = {"python": platform.python_version()} | {name: version(name) for name in libraries}
    (out / "meta.json").write_text(json.dumps(meta | {"versions": versions}, indent=2))
    print(f"saved {family} to {out}")
