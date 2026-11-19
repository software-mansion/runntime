import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from safetensors.numpy import save_file

from runntime_tools.safetensors_io import SHARD_INDEX, Tensor, read_checkpoint, to_float32


@pytest.mark.parametrize(
    ("dtype", "data"),
    [
        ("F32", np.array([1, -2], dtype="<f4")),
        ("F16", np.array([0x3C00, 0xC000], dtype="<u2")),
        ("BF16", np.array([0x3F80, 0xC000], dtype="<u2")),
    ],
)
def test_to_float32_decodes(dtype: str, data: NDArray[np.generic]) -> None:
    tensor = Tensor(dtype, (2,), data.view(np.uint8))
    np.testing.assert_array_equal(to_float32(tensor), [1, -2])


def test_read_checkpoint_joins_the_shards_of_a_folder(tmp_path: Path) -> None:
    save_file({"a": np.ones(2, dtype=np.float32)}, tmp_path / "shard-1.safetensors", {"format": "pt"})
    save_file({"b": np.zeros(3, dtype=np.float16)}, tmp_path / "shard-2.safetensors")
    weight_map = {"a": "shard-1.safetensors", "b": "shard-2.safetensors"}
    (tmp_path / SHARD_INDEX).write_text(json.dumps({"weight_map": weight_map}))

    checkpoint = read_checkpoint(tmp_path)

    assert list(checkpoint.tensors) == ["a", "b"]
    np.testing.assert_array_equal(to_float32(checkpoint.tensors["b"]), [0, 0, 0])
    assert checkpoint.metadata == {"format": "pt"}
