import argparse
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from safetensors import safe_open

from runntime_tools.cli import Options, group_size, main, skip_reason
from runntime_tools.quantize import Bits
from runntime_tools.safetensors_io import Tensor, from_array, write_safetensors

GOLDEN = Path(__file__).parent / "golden"
"""tiny.safetensors and what `runntime quantize --group-size 32` makes of it
at 4 and 8 bits. The runntime package's TypeScript tests read the same files."""


@pytest.mark.parametrize("bits", [4, 8])
def test_quantize_writes_the_golden_file(bits: Bits, tmp_path: Path) -> None:
    destination = tmp_path / "out.safetensors"
    argv = ["quantize", str(GOLDEN / "tiny.safetensors"), str(destination), "--bits", str(bits), "--group-size", "32"]
    assert main(argv) == 0
    assert destination.read_bytes() == (GOLDEN / f"tiny.int{bits}.safetensors").read_bytes()


def test_quantize_can_overwrite_its_source(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes((GOLDEN / "tiny.safetensors").read_bytes())
    assert main(["quantize", str(checkpoint), str(checkpoint), "--bits", "8", "--group-size", "32"]) == 0
    assert checkpoint.read_bytes() == (GOLDEN / "tiny.int8.safetensors").read_bytes()
    assert list(tmp_path.iterdir()) == [checkpoint]


def test_golden_file_reads_back_with_the_safetensors_library() -> None:
    with (
        safe_open(GOLDEN / "tiny.int4.safetensors", "numpy") as quantized,
        safe_open(GOLDEN / "tiny.safetensors", "numpy") as source,
    ):
        metadata = quantized.metadata()
        assert metadata["format"] == "pt"
        assert metadata["runntime.quant"] == '{"version":1,"scheme":"symmetric"}'
        weight = quantized.get_tensor("layer.weight")
        assert (weight.dtype, weight.shape) == (np.uint8, (8, 32))
        scale = quantized.get_tensor("layer.weight_scale")
        assert (scale.dtype, scale.shape) == (np.float16, (8, 2))
        for name in ("layer.bias", "norm.weight", "odd.weight"):
            np.testing.assert_array_equal(quantized.get_tensor(name), source.get_tensor(name))


@pytest.mark.parametrize(("bits", "dtype", "stored_cols"), [(8, np.int8, 64), (4, np.uint8, 32)])
def test_golden_file_stores_a_conv_weight_as_rows(bits: Bits, dtype: type, stored_cols: int) -> None:
    # conv.weight is [out 6, in 16, kernel 4]: rows of 16 * 4 = 64 values, two groups of 32.
    with (
        safe_open(GOLDEN / f"tiny.int{bits}.safetensors", "numpy") as quantized,
        safe_open(GOLDEN / "tiny.safetensors", "numpy") as source,
    ):
        stored = quantized.get_tensor("conv.weight")
        assert (stored.dtype, stored.shape) == (dtype, (6, stored_cols))
        scales = quantized.get_tensor("conv.weight_scale")
        assert (scales.dtype, scales.shape) == (np.float16, (6, 2))
        ints = stored if bits == 8 else unpack_int4(stored)
        # Row-major, so value (o, c, k) of the conv is value c * 4 + k of row o.
        values = (ints.reshape(6, 2, 32) * scales.astype(np.float32)[..., np.newaxis]).reshape(6, 16, 4)
        step = np.repeat(scales.astype(np.float32), 32, axis=1).reshape(6, 16, 4)
        assert np.all(np.abs(values - source.get_tensor("conv.weight")) <= step / 2 + 1e-6)


def unpack_int4(packed: NDArray[np.uint8]) -> NDArray[np.int8]:
    """Two signed nibbles per byte, the low one first, as one value each."""
    nibbles = np.stack([packed & 0x0F, packed >> 4], axis=-1).reshape(packed.shape[0], -1).astype(np.int8)
    return np.where(nibbles > 7, nibbles - 16, nibbles).astype(np.int8)


def tensor(shape: tuple[int, ...], dtype: str = "F32") -> Tensor:
    return Tensor(dtype, shape, np.zeros(0, dtype=np.uint8))


@pytest.mark.parametrize(
    ("name", "stored", "bits", "group_size", "reason"),
    [
        ("fc.weight", tensor((8, 64)), 4, 32, None),
        ("fc.weight", tensor((4, 64), "BF16"), 8, 32, None),
        ("fc.weight", tensor((8, 64), "F16"), 8, 32, None),
        # Shapes no layer rule looks at: any out, any row the group size divides.
        ("fc.weight", tensor((1, 32)), 4, 32, None),
        ("fc.weight", tensor((5, 96)), 8, 32, None),
        ("fc.weight", tensor((7, 36)), 8, 12, None),
        ("fc.weight", tensor((3, 5)), 8, 1, None),
        ("fc.weight", tensor((3, 9)), 8, 3, None),
        # 3 or more dims quantize as rows of everything after dim 0.
        ("conv.weight", tensor((8, 64, 3)), 8, 32, None),
        ("conv.weight", tensor((8, 16, 3, 3)), 4, 16, None),
        ("conv.weight", tensor((5, 3, 7)), 8, 7, None),
        ("fc.bias", tensor((8,)), 8, 32, "not matched by --include"),
        ("skip.weight", tensor((8, 64)), 8, 32, "excluded by skip.*"),
        ("fc.weight", tensor((8, 64), "I8"), 8, 32, "dtype I8"),
        ("fc.weight", tensor((8, 64), "F64"), 8, 32, "dtype F64"),
        ("norm.weight", tensor((64,)), 8, 32, "1D, needs 2 or more dims"),
        ("scale.weight", tensor(()), 8, 1, "0D, needs 2 or more dims"),
        ("fc.weight", tensor((8, 48)), 8, 32, "row of 48 values not a multiple of the group size 32"),
        ("conv.weight", tensor((8, 16, 3)), 8, 32, "row of 48 values not a multiple of the group size 32"),
        ("fc.weight", tensor((8, 9)), 4, 3, "row of 9 values is odd, int4 packs two to a byte"),
    ],
)
def test_skip_reason(name: str, stored: Tensor, bits: Bits, group_size: int, reason: str | None) -> None:
    options = Options(bits=bits, group_size=group_size, include=("*.weight",), exclude=("skip.*",))
    assert skip_reason(name, stored, options) == reason


def test_quantizes_embeddings_and_lm_head_by_default(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "model.safetensors"
    weight = from_array("F32", np.ones((4, 32), dtype=np.float32))
    names = ("embed_tokens.weight", "pos_emb.weight", "lm_head.weight")
    write_safetensors(source, dict.fromkeys(names, weight), {})
    assert main(["quantize", str(source), str(tmp_path / "out.safetensors"), "--bits", "8"]) == 0
    assert [line.split()[0] for line in capsys.readouterr().out.splitlines()] == ["quantized"] * 3


@pytest.mark.parametrize("size", ["1", "3", "6", "640"])
def test_group_size_takes_any_positive_integer(size: str) -> None:
    assert group_size(size) == int(size)


@pytest.mark.parametrize("size", ["0", "-4"])
def test_group_size_refuses_zero_and_negatives(size: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        group_size(size)


def test_report_names_the_stored_shape_of_a_conv_weight(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["quantize", str(GOLDEN / "tiny.safetensors"), str(tmp_path / "out.safetensors"), "--bits", "8"]
    assert main(argv) == 0
    assert "quantized  conv.weight [6, 16, 4] as [6, 64]: max error" in capsys.readouterr().out


def test_quantize_fails_when_no_tensor_is_selected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    destination = tmp_path / "out.safetensors"
    argv = ["quantize", str(GOLDEN / "tiny.safetensors"), str(destination), "--bits", "8", "--include", "*.bias"]
    assert main(argv) == 1
    assert "no tensor was quantized" in capsys.readouterr().err
    assert not destination.exists()


def test_quantize_fails_on_a_quantized_source(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    destination = tmp_path / "out.safetensors"
    argv = ["quantize", str(GOLDEN / "tiny.int8.safetensors"), str(destination), "--bits", "4"]
    assert main(argv) == 1
    assert "is already quantized" in capsys.readouterr().err
    assert not destination.exists()


def test_f16_stores_the_float_tensors_that_stay_as_f16(tmp_path: Path) -> None:
    destination = tmp_path / "out.safetensors"
    argv = ["quantize", str(GOLDEN / "tiny.safetensors"), str(destination), "--bits", "8", "--f16"]
    assert main(argv) == 0
    with (
        safe_open(destination, "numpy") as quantized,
        safe_open(GOLDEN / "tiny.safetensors", "numpy") as source,
    ):
        assert quantized.get_tensor("layer.weight").dtype == np.int8
        for name in ("layer.bias", "norm.weight", "odd.weight"):
            stored = quantized.get_tensor(name)
            assert stored.dtype == np.float16
            np.testing.assert_array_equal(stored, source.get_tensor(name).astype(np.float16))


def test_f16_fails_on_a_value_beyond_the_f16_range(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "big.safetensors"
    weight = np.zeros((8, 32), dtype=np.float32)
    bias = np.array([1e5], dtype=np.float32)
    write_safetensors(source, {"fc.weight": from_array("F32", weight), "fc.bias": from_array("F32", bias)}, {})
    assert main(["quantize", str(source), str(tmp_path / "out.safetensors"), "--bits", "8", "--f16"]) == 1
    assert "fc.bias has values beyond the f16 range" in capsys.readouterr().err


def test_drop_leaves_matching_tensors_out(tmp_path: Path) -> None:
    destination = tmp_path / "out.safetensors"
    argv = ["quantize", str(GOLDEN / "tiny.safetensors"), str(destination), "--bits", "8", "--drop", "norm.*"]
    assert main(argv) == 0
    with safe_open(destination, "numpy") as quantized:
        dropped = {"norm.weight"}
        scales = {"conv.weight_scale", "embed.weight_scale", "layer.weight_scale"}
        with safe_open(GOLDEN / "tiny.safetensors", "numpy") as source:
            assert set(quantized.keys()) == set(source.keys()) - dropped | scales
