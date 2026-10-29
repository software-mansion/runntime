"""Reads and writes safetensors files as raw bytes, so every dtype (BF16 too)
copies through unchanged and only the tensors that get quantized are decoded."""

import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

HEADER_LENGTH_BYTES = 8
"""The file starts with the JSON header's length as a little-endian u64."""

HEADER_ALIGNMENT_BYTES = 8
"""The JSON header is padded with spaces to this many bytes."""

SHARD_INDEX = "model.safetensors.index.json"
"""Names the shards of a Hugging Face checkpoint folder."""

SINGLE_FILE = "model.safetensors"
"""The weights of a Hugging Face checkpoint folder that is not sharded."""


@dataclass(frozen=True)
class Tensor:
    """A tensor as stored: its safetensors dtype, shape and raw bytes."""

    dtype: str
    shape: tuple[int, ...]
    data: NDArray[np.uint8]


@dataclass(frozen=True)
class Checkpoint:
    """Every tensor of a checkpoint in file order, and its header metadata."""

    tensors: dict[str, Tensor]
    metadata: dict[str, str]


def read_checkpoint(path: Path) -> Checkpoint:
    """Reads one safetensors file, or a Hugging Face folder: its sharded
    weights when it has a shard index, else its `model.safetensors`. Tensor
    bytes are memory-mapped, not loaded."""
    if path.is_file():
        return _read_file(path)
    index = path / SHARD_INDEX
    if not index.exists():
        return _read_file(path / SINGLE_FILE)
    weight_map: dict[str, str] = json.loads(index.read_text())["weight_map"]
    tensors: dict[str, Tensor] = {}
    metadata: dict[str, str] = {}
    for shard in dict.fromkeys(weight_map.values()):
        checkpoint = _read_file(path / shard)
        tensors |= checkpoint.tensors
        metadata |= checkpoint.metadata
    return Checkpoint(tensors, metadata)


def _read_file(path: Path) -> Checkpoint:
    with path.open("rb") as file:
        (header_length,) = struct.unpack("<Q", file.read(HEADER_LENGTH_BYTES))
        header: dict[str, Any] = json.loads(file.read(header_length))
    metadata: dict[str, str] = header.pop("__metadata__", {})
    payload = np.memmap(path, dtype=np.uint8, mode="r", offset=HEADER_LENGTH_BYTES + header_length)
    tensors: dict[str, Tensor] = {}
    for name, info in header.items():
        begin, end = info["data_offsets"]
        tensors[name] = Tensor(info["dtype"], tuple(info["shape"]), payload[begin:end])
    return Checkpoint(tensors, metadata)


def write_safetensors(path: Path, tensors: Mapping[str, Tensor], metadata: Mapping[str, str]) -> None:
    """Writes `tensors` in their order, with `metadata` in the header."""
    header: dict[str, object] = {"__metadata__": dict(metadata)} if metadata else {}
    offset = 0
    for name, tensor in tensors.items():
        header[name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + tensor.data.nbytes],
        }
        offset += tensor.data.nbytes
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % HEADER_ALIGNMENT_BYTES)
    with path.open("wb") as file:
        file.write(struct.pack("<Q", len(encoded)))
        file.write(encoded)
        for tensor in tensors.values():
            file.write(tensor.data.tobytes())


def to_float32(tensor: Tensor) -> NDArray[np.float32]:
    """Decodes an F32, F16 or BF16 tensor to f32 in its shape."""
    match tensor.dtype:
        case "F32":
            values = tensor.data.view("<f4").astype(np.float32)
        case "F16":
            values = tensor.data.view("<f2").astype(np.float32)
        case "BF16":
            # bf16 is the top half of an f32's bits.
            values = (tensor.data.view("<u2").astype(np.uint32) << 16).view(np.float32)
        case _:
            raise ValueError(f"cannot decode {tensor.dtype} to f32")
    return values.reshape(tensor.shape)


def from_array(dtype: str, values: NDArray[np.generic]) -> Tensor:
    """A tensor over the little-endian bytes of `values`."""
    data = np.ascontiguousarray(values).reshape(-1).view(np.uint8)
    return Tensor(dtype, tuple(values.shape), data)
