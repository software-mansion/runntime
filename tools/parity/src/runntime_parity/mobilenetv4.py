"""MobileNetV4-Conv-S reference: the logits and the top class of the timm
model in f32, on COCO images through timm's own preprocessing.

    uv run python -m runntime_parity.mobilenetv4
"""

import numpy as np
import timm
import torch
from numpy.typing import NDArray

from .coco import ATTRIBUTION, coco_images
from .common import cosine_noise, devices, noise, save

REPO = "timm/mobilenetv4_conv_small.e2400_r224_in1k"
REVISION = "331fb803779522b685cf942e15f914fb6741c1eb"


def centered(logits: NDArray[np.float32]) -> NDArray[np.float32]:
    """Logits with each row's mean subtracted, as the parity test compares them."""
    return logits - logits.mean(axis=1, keepdims=True)


def main() -> None:
    paths, images, sources = coco_images()
    model = timm.create_model(f"hf_hub:{REPO}@{REVISION}", pretrained=True).eval()
    transform = timm.data.create_transform(**timm.data.resolve_model_data_config(model), is_training=False)
    pixels = torch.stack([transform(image) for image in images])

    def logits(device: str) -> NDArray[np.float32]:
        with torch.inference_mode():
            return model.to(device)(pixels.to(device)).cpu().numpy()

    device, noise_device = devices()
    reference, second = logits(device), logits(noise_device)
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
            "inputs": ATTRIBUTION | {"images": sources},
            "devices": [device, noise_device],
            "noise": {
                "error": noise(reference, second),
                "cosineDistance": cosine_noise(centered(reference), centered(second)),
            },
        },
        ["torch", "timm", "torchvision", "pillow"],
    )


if __name__ == "__main__":
    main()
