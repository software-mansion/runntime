from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open

from runntime_tools.cli import Options, main, skip_reason
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
        assert metadata["runntime.quant"] == '{"version":1,"bits":4,"groupSize":32,"scheme":"symmetric"}'
        weight = quantized.get_tensor("layer.weight")
        assert (weight.dtype, weight.shape) == (np.uint8, (8, 32))
        scale = quantized.get_tensor("layer.weight_scale")
        assert (scale.dtype, scale.shape) == (np.float16, (8, 2))
        for name in ("embed.weight", "layer.bias", "norm.weight"):
            np.testing.assert_array_equal(quantized.get_tensor(name), source.get_tensor(name))


def tensor(shape: tuple[int, ...], dtype: str = "F32") -> Tensor:
    return Tensor(dtype, shape, np.zeros(0, dtype=np.uint8))


@pytest.mark.parametrize(
    ("name", "stored", "bits", "reason"),
    [
        ("fc.weight", tensor((8, 64)), 4, None),
        ("fc.weight", tensor((4, 64), "BF16"), 8, None),
        ("fc.bias", tensor((8,)), 8, "not matched by --include"),
        ("embed.weight", tensor((8, 64)), 8, "excluded by *embed*"),
        ("fc.weight", tensor((8, 64), "I8"), 8, "dtype I8"),
        ("conv.weight", tensor((8, 64, 3)), 8, "3D, needs [out, in]"),
        ("conv.weight", tensor((8, 64, 3, 3)), 8, "4D, needs [out, in]"),
        ("fc.weight", tensor((8, 36)), 8, "in features 36 not a multiple of 8"),
        ("fc.weight", tensor((8, 48)), 8, "in features 48 not a multiple of the group size 32"),
        ("fc.weight", tensor((4, 64)), 4, "out features 4 not a multiple of 8"),
        ("fc.weight", tensor((6, 64)), 8, "out features 6 not a multiple of 4"),
    ],
)
def test_skip_reason(name: str, stored: Tensor, bits: Bits, reason: str | None) -> None:
    options = Options(bits=bits, group_size=32, include=("*.weight",), exclude=("*embed*",))
    assert skip_reason(name, stored, options) == reason


def test_quantize_fails_when_no_tensor_is_selected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    destination = tmp_path / "out.safetensors"
    argv = ["quantize", str(GOLDEN / "tiny.safetensors"), str(destination), "--bits", "8", "--include", "*.bias"]
    assert main(argv) == 1
    assert "no tensor was quantized" in capsys.readouterr().err
    assert not destination.exists()


def test_quantize_fails_on_a_quantized_source(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    destination = tmp_path / "out.safetensors"
    argv = ["quantize", str(GOLDEN / "tiny.int8.safetensors"), str(destination), "--bits", "4", "--exclude", "none"]
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
        for name in ("embed.weight", "layer.bias", "norm.weight"):
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
        assert set(quantized.keys()) == {"embed.weight", "layer.bias", "layer.weight", "layer.weight_scale"}
