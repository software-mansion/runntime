"""What every reference script shares: where references go, the reference
noise, and the files a run saves."""

import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from safetensors.numpy import save_file


def references_dir(family: str) -> Path:
    """The folder the parity tests read `family` from: `RUNNTIME_PARITY_DIR`,
    or `~/.cache/runntime-parity`."""
    root = os.environ.get("RUNNTIME_PARITY_DIR") or Path.home() / ".cache" / "runntime-parity"
    path = Path(root) / family
    path.mkdir(parents=True, exist_ok=True)
    return path


def noise(reference: NDArray[np.float32], double: NDArray[np.float64]) -> float:
    """The largest difference between the reference and the same model run
    in f64."""
    return float(np.abs(reference.astype(np.float64) - double).max())


def cosine_noise(reference: NDArray[np.float32], double: NDArray[np.float64]) -> float:
    """The largest cosine distance between the rows of the reference and of
    the same model run in f64."""
    a, b = reference.astype(np.float64), double
    cosine = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return float((1 - cosine).max())


def save(
    family: str,
    inputs: object,
    tensors: dict[str, NDArray[np.float32] | NDArray[np.int32]],
    meta: dict[str, Any],
    libraries: list[str],
) -> None:
    """Writes `inputs.json`, `reference.safetensors` and `meta.json`, with the
    version of Python and of each library in `libraries`. Refuses a noise of
    0 in `meta`: a threshold in multiples of it is 0."""
    for name, value in meta["noise"].items():
        if value == 0:
            raise RuntimeError(f"the reference noise of {name} is 0")
    out = references_dir(family)
    (out / "inputs.json").write_text(json.dumps(inputs, ensure_ascii=False, indent=2))
    save_file(tensors, out / "reference.safetensors")
    versions = {"python": platform.python_version()} | {name: version(name) for name in libraries}
    (out / "meta.json").write_text(json.dumps(meta | {"versions": versions}, indent=2))
    print(f"saved {family} to {out}")
