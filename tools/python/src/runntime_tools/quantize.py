"""Symmetric group-wise int8/int4 quantization of [out, in] weights, in the
runntime format v1 layout."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Bits = Literal[4, 8]


@dataclass(frozen=True)
class QuantizedWeight:
    """`ints` [out, in], one signed value per element, and `scales`
    [out, in / group_size], one f16 scale per group of inputs."""

    ints: NDArray[np.int8]
    scales: NDArray[np.float16]


def quantize(weight: NDArray[np.float32], bits: Bits, group_size: int) -> QuantizedWeight:
    """Quantizes each group of `group_size` inputs of each output row to
    `int = round(w / scale)` with `scale = absmax / (2^(bits-1) - 1)`, so the
    range is symmetric and has no zero point. The ints divide by the scale
    rounded to f16, the one the file stores. An all-zero group has scale 0
    and ints 0."""
    out_features, in_features = weight.shape
    largest = 2 ** (bits - 1) - 1
    groups = weight.reshape(out_features, in_features // group_size, group_size)
    scales = (np.abs(groups).max(axis=-1, keepdims=True) / largest).astype(np.float16)
    divisor = scales.astype(np.float32)
    ratio = np.divide(groups, divisor, out=np.zeros_like(groups), where=divisor != 0)
    ints = np.clip(np.rint(ratio), -largest, largest).astype(np.int8)
    return QuantizedWeight(ints.reshape(weight.shape), scales.squeeze(-1))


def dequantize(weight: QuantizedWeight) -> NDArray[np.float32]:
    """The float values the ints stand for: int * scale."""
    out_features, in_features = weight.ints.shape
    group_count = weight.scales.shape[1]
    groups = weight.ints.reshape(out_features, group_count, in_features // group_count)
    values = groups * weight.scales.astype(np.float32)[..., np.newaxis]
    return values.reshape(out_features, in_features)


def pack_int4(ints: NDArray[np.int8]) -> NDArray[np.uint8]:
    """Packs signed int4 values two to a byte along the last axis: the even
    index in the low nibble, each as a 4-bit two's complement."""
    nibbles = ints.astype(np.uint8) & np.uint8(0x0F)
    return nibbles[..., 0::2] | (nibbles[..., 1::2] << np.uint8(4))
