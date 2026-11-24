"""DepthART reference: the relative depth map of the upstream model in f32,
on COCO images. The preprocessing starts from the decoded RGB pixels coco.py
shares with the test, then follows upstream's relative/infer_image.py. One
reference per size.

    uv run python -m runntime_parity.depthart s
    uv run python -m runntime_parity.depthart b

The weights are read from the `depthart` folder of the references directory,
`relative_<size>_448.safetensors`. Each is `relative/depthart_relative_<size>_448.pth`
of Fengxue93/DepthART at REVISION, checked against the repo's CHECKSUMS.sha256,
loaded with `torch.load(weights_only=True)` in a separate venv, so this one
never unpickles, and saved with safetensors, its tensors unchanged.
"""

import importlib
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from safetensors.torch import load as load_tensors

from .coco import ATTRIBUTION, coco_images
from .common import checked, cosine_noise, noise, references_dir, save

REPO = "Fengxue93/DepthART"
REVISION = "9e154201cd8fb3453c9def581997aaf1b4da3c22"
# The upstream model code: xuefeng-cvr/DepthART on GitHub at this commit.
CODE = "b307c123f19051e2d969960bba93ee3684874b90"
CODE_FILES = f"https://raw.githubusercontent.com/xuefeng-cvr/DepthART/{CODE}/relative"
# The files of it the model needs, by their path under `relative`, each
# with its sha256.
CODE_SHA256 = {
    "models.py": "1246f83c759ee647c19e1e4857e71babf40dbd9aa9cb500037e427016ccdca93",
    "tinyvim/model/__init__.py": "4d0eceea415a0cbddb08fdaa16b9b2469571ee5fe26bfa3cecfa923735c822bb",
    "tinyvim/model/dpt.py": "bc118ae44aaf74a95638d43e962569df110db9acb736f4a2716c01c3ebd99df7",
    "tinyvim/model/tinyvim.py": "50b0afe1accd16036f37787939bf0afb355f0526e3c091aef441ea06053337d1",
    "tinyvim/model/tvimblock.py": "01265a12a37d0bd3a18110cc1288a09f4b3a57ad8b8a9cd30bf014fb71eddae1",
    "tinyvim/model/util/blocks.py": "351aa24b8af4bef923ebe3bed5e5e8503b08e39bf06ee045d6444a485a2036a8",
}
# The weights as safetensors, by size, each with its sha256.
WEIGHTS_SHA256 = {
    "s": "faed07def1d6d1106aeb89f5f9268c78b9b79fa938fb56e19ae078975a834009",
    "b": "de2c6363027557b1569f735cb6c7d7bae7cbf2ad16ab48dd29f53915c58ab013",
}

SIZE = 448
# As relative/infer_image.py of the upstream code normalizes.
MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def upstream_code() -> Path:
    """The `relative` folder of the upstream code with the model's files,
    each checked against its sha256. Nothing of it is built or installed."""
    root = references_dir("depthart-source") / CODE / "relative"
    for name, sha256 in CODE_SHA256.items():
        path = root / name
        if path.exists():
            checked(str(path), path.read_bytes(), sha256)
            continue
        url = f"{CODE_FILES}/{name}"
        with urllib.request.urlopen(url) as response:
            data = checked(url, response.read(), sha256)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


def load(size: str) -> torch.nn.Module:
    """The upstream model of `size` on the CPU, in f32. Without upstream's
    CUDA extension its selective scan is the portable one."""
    sys.path.insert(0, str(upstream_code()))
    model = importlib.import_module("models").create_relative_model(size.upper())
    weights = references_dir("depthart") / f"relative_{size}_448.safetensors"
    model.load_state_dict(load_tensors(checked(str(weights), weights.read_bytes(), WEIGHTS_SHA256[size])), strict=True)
    return model.eval()


def preprocess(image: Image.Image) -> NDArray[np.float32]:
    """The model input of an image's RGB pixels, as relative/infer_image.py
    makes it from them."""
    rgb = np.asarray(image).astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    return ((rgb - MEAN) / STD).transpose(2, 0, 1).copy()


def centered(maps: NDArray) -> NDArray[np.float64]:
    """Maps as rows with each row's mean subtracted, as the parity test
    compares them."""
    rows = maps.reshape(len(maps), -1).astype(np.float64)
    return rows - rows.mean(axis=1, keepdims=True)


def main() -> None:
    size = sys.argv[1]
    paths, images, sources = coco_images()
    pixels = torch.from_numpy(np.stack([preprocess(image) for image in images]))
    model = load(size)

    with torch.inference_mode():
        reference = model(pixels)[:, 0].numpy()
        # The same model and pixels in f64 give the reference noise. The
        # selective scan computes in f32 whatever it is given, so the noise
        # is f32 against mostly f64.
        double = model.double()(pixels.double())[:, 0].numpy()
    save(
        f"depthart-{size}",
        {"images": paths},
        {"pixels": pixels.numpy(), "depth": reference},
        {
            "repo": REPO,
            "revision": REVISION,
            "license": "CC BY-NC 4.0",
            "dtype": "f32",
            "inputs": ATTRIBUTION | {"count": len(images), "images": sources},
            "settings": {"size": size, "code": CODE},
            "noise": {
                "error": noise(reference, double),
                "cosineDistance": cosine_noise(centered(reference), centered(double)),
            },
        },
        ["torch", "timm", "einops", "opencv-python-headless", "pillow"],
    )


if __name__ == "__main__":
    main()
