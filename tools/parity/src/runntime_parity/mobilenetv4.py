"""MobileNetV4-Conv-S reference: the logits and the top class of the timm
model in f32, on COCO images through timm's own preprocessing.

    uv run python -m runntime_parity.mobilenetv4
"""

import numpy as np
import timm
import torch
from numpy.typing import NDArray

from .coco import ATTRIBUTION, coco_images
from .common import cosine_noise, noise, save

REPO = "timm/mobilenetv4_conv_small.e2400_r224_in1k"
REVISION = "331fb803779522b685cf942e15f914fb6741c1eb"


def centered(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Logits with each row's mean subtracted, as the parity test compares them."""
    return logits - logits.mean(axis=1, keepdims=True)


def main() -> None:
    paths, images, sources = coco_images()
    model = timm.create_model(f"hf_hub:{REPO}@{REVISION}", pretrained=True).eval()
    transform = timm.data.create_transform(**timm.data.resolve_model_data_config(model), is_training=False)
    pixels = torch.stack([transform(image) for image in images])

    with torch.inference_mode():
        reference = model(pixels).numpy()
        # The same model and pixels in f64 give the reference noise.
        double = model.double()(pixels.double()).numpy()
    save(
        "mobilenetv4",
        {"images": paths},
        {
            "pixels": pixels.numpy(),
            "logits": reference,
            "top1": reference.argmax(axis=1).astype(np.int32),
        },
        {
            "repo": REPO,
            "revision": REVISION,
            "dtype": "f32",
            "inputs": ATTRIBUTION | {"count": len(images), "images": sources},
            "noise": {
                "error": noise(reference, double),
                "cosineDistance": cosine_noise(centered(reference.astype(np.float64)), centered(double)),
            },
        },
        ["torch", "timm", "torchvision", "pillow"],
    )


if __name__ == "__main__":
    main()
