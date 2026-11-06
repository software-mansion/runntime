import numpy as np
import pytest

from runntime_tools.quantize import Bits, QuantizedWeight, dequantize, pack_int4, quantize


# Each group's absmax equals the top int, so the scale is 1 and exact in f16.
# The halves show the rounding: half to even.
@pytest.mark.parametrize(
    ("bits", "row", "ints"),
    [
        (8, [127, -63.5, 0, 1.4], [127, -64, 0, 1]),
        (4, [7, -3.5, 0, 2.5], [7, -4, 0, 2]),
    ],
)
def test_quantize_rounds_to_the_nearest_step(bits: Bits, row: list[float], ints: list[int]) -> None:
    weight = np.array([row], dtype=np.float32)
    quantized = quantize(weight, bits, group_size=4)
    np.testing.assert_array_equal(quantized.ints, [ints])
    np.testing.assert_array_equal(quantized.scales, [[1]])


def test_quantize_gives_each_group_its_own_scale() -> None:
    # Groups of 4: absmax 254 → scale 2, absmax 0 → scale 0 and ints 0.
    weight = np.array([[254, -2, 4, 0, 0, 0, 0, 0]], dtype=np.float32)
    quantized = quantize(weight, bits=8, group_size=4)
    np.testing.assert_array_equal(quantized.ints, [[127, -1, 2, 0, 0, 0, 0, 0]])
    np.testing.assert_array_equal(quantized.scales, [[2, 0]])


@pytest.mark.parametrize("bits", [4, 8])
def test_dequantized_values_are_within_half_a_step(bits: Bits) -> None:
    weight = np.random.default_rng(0).standard_normal((16, 64)).astype(np.float32)
    quantized = quantize(weight, bits, group_size=32)
    step = np.repeat(quantized.scales.astype(np.float32), 32, axis=1)
    assert np.all(np.abs(dequantize(quantized) - weight) <= step / 2 + 1e-6)


def test_dequantize_multiplies_each_int_by_its_group_scale() -> None:
    ints = np.array([[1, -2, 3, 4]], dtype=np.int8)
    scales = np.array([[0.5, 2]], dtype=np.float16)
    np.testing.assert_array_equal(dequantize(QuantizedWeight(ints, scales)), [[0.5, -1, 6, 8]])


def test_pack_int4_puts_the_even_index_in_the_low_nibble() -> None:
    # 1, -1 → 0xf1. 7, -7 → 0x97. -8, 0 → 0x08.
    ints = np.array([[1, -1, 7, -7, -8, 0]], dtype=np.int8)
    np.testing.assert_array_equal(pack_int4(ints), [[0xF1, 0x97, 0x08]])
