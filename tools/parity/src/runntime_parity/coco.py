"""The images every vision reference runs on: COCO val2017 photos under
CC BY 2.0 from the COCO site, saved once as PNG so the reference and the
browser read the same pixels."""

import hashlib
import io
import json
import urllib.request
import zipfile

import numpy as np
from PIL import Image

from .common import references_dir

# COCO 2017 (Lin et al., 2014). The site serves plain HTTP only, so every
# file is checked against its sha256.
SITE = "http://images.cocodataset.org"
ANNOTATIONS_ZIP = f"{SITE}/annotations/annotations_trainval2017.zip"
# The smallest file in the zip that lists every val2017 image with its license.
ANNOTATIONS = "annotations/captions_val2017.json"
ANNOTATIONS_SHA256 = "afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9"
# The id of CC BY 2.0 in the annotations' license list.
CC_BY = 4
# The first 25 CC BY images by id, each with the sha256 of its JPEG.
IMAGES = {
    285: "f3a2974ce3686332609124c70e3e6a2e3aca43fccf1cd1bd7c5c03820977f57d",
    785: "83981537a7baeafbeb9c8cb67b3484dc26433f574b3685d021fa537e277e4726",
    872: "c2aa138ee3a59b057a7ba6fc5a6a18e62af531aa7dab78a7bfd33c1cd7e55eb6",
    885: "1c67b783f78b18ddb5dd20aaba4d4aacaf98e5277051688a96da096afa9cdf1f",
    1000: "24bb77a31928404e45a0454b06f6a0bd54a8db103590c9d7917288a1e0269f05",
    1268: "8e85a2e71ee512fe748a3a3be9bb40b8591e8c501c0f3896efcb5be5999ec236",
    1993: "039f3bfb862023bde7f835a36c32d8f59a3e66488863c2a47b7b746e78c44186",
    2473: "2f1bfa7634a64f7999ba5488ec904ad1f85bbc63d0fe4b9fe35f52abbf8c5f4f",
    5001: "99600c56dc25c72dc1cc1d1f2b09aa33b97142560b54d034fa83a38fcf3c5b0f",
    6012: "7cd0d627b15c09f373aa613d013fb2d8ae6bd20bcce0d98aa31b963e6bcca495",
    6818: "99479f61ef2931ea70e161db0859e8706c6e48f13e8cb0798959f52433b01b98",
    7281: "57f3508e600d3370320487dde8665f208f11612a11caac0801c7f5fc19188ac3",
    7574: "4f26c847c2f49574978e96733efaec720dd1d1ba95a8220d0de012b7317eed47",
    7991: "63c6fae2e3defd4bc1c6f346806aef03eb11ccbd7c6bda3e479d1fbafd486db1",
    8021: "235537407867a7b894e78b3aecb55b33b14b64cbb23ca376a00d34c62a1a63a6",
    8762: "8b8aa2412ac3502dbb4b8a7d7987359d4049cf70e7177748e907e3ddfa3211ce",
    9483: "c7e6e15f4db067c3e09ab8605759be9d850e2d5c45ea4dfb9cda8f5d85308ea4",
    11149: "5973b8f1d5a30de98b1dcb715f1c94928a676fff8e45283515d0474e51ac0abc",
    12062: "319d58f0fa008d076b766671a5b607f94614f6005f773adced54a57567aa28fb",
    12120: "5078da2ea4554eb91703e5fa0339c4d0c875ded1783a8f4ee32ece2b558178c4",
    12280: "158ef0b15ff392039468263fa18735f81b39de9e8a7753e1e1aba7037262da56",
    12667: "0bf182b59b89f534d6f89a2b57c976c47b7f9a95e5e47a7f236b285b9211af14",
    13004: "09d489992329d2a17186a82781c7d23d8c687ee580567babc7494f9bc8b1220a",
    13291: "1e46187ec49947bfd8dd22f4d774bddfe3e29ca863fe09530516598998b8e640",
    13348: "8371f759628048bd294db70d9fbacb6d4f2558a12c25337fa04fea6418c9922a",
}

ATTRIBUTION = {
    "dataset": "COCO val2017",
    "authors": "Lin et al., 2014",
    "source": SITE,
    "annotations": {"file": ANNOTATIONS, "sha256": ANNOTATIONS_SHA256},
    "license": "CC BY 2.0",
}


class _HttpFile(io.RawIOBase):
    """A file at a URL, read in ranges: zipfile takes one member out of it
    without the rest of the archive."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.position = 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as response:
            self.size = int(response.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.position = (0, self.position, self.size)[whence] + offset
        return self.position

    def readinto(self, buffer: bytearray) -> int:
        end = min(self.position + len(buffer), self.size)
        if end <= self.position:
            return 0
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.position}-{end - 1}"})
        with urllib.request.urlopen(request) as response:
            data = response.read()
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


def _checked(what: str, data: bytes, sha256: str) -> bytes:
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise RuntimeError(f"{what} has sha256 {digest}, expected {sha256}")
    return data


def coco_images() -> tuple[list[str], list[Image.Image], list[dict[str, object]]]:
    """The first CC BY images by id: each one's path under the references
    directory, its pixels, and where it comes from."""
    with zipfile.ZipFile(io.BufferedReader(_HttpFile(ANNOTATIONS_ZIP), buffer_size=1 << 20)) as archive:
        listed = json.loads(_checked(ANNOTATIONS, archive.read(ANNOTATIONS), ANNOTATIONS_SHA256))["images"]
    chosen = sorted((image for image in listed if image["license"] == CC_BY), key=lambda image: image["id"])
    chosen = chosen[: len(IMAGES)]
    if [image["id"] for image in chosen] != list(IMAGES):
        raise RuntimeError("IMAGES is not the first CC BY images of the annotations")
    out = references_dir("coco")
    paths, images, sources = [], [], []
    for entry in chosen:
        url = f"{SITE}/val2017/{entry['file_name']}"
        with urllib.request.urlopen(url) as response:
            jpeg = _checked(url, response.read(), IMAGES[entry["id"]])
        # From the bare pixels, so the PNG carries no color profile.
        image = Image.fromarray(np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB")))
        name = f"{entry['id']}.png"
        image.save(out / name)
        paths.append(f"coco/{name}")
        images.append(image)
        sources.append({"id": entry["id"], "url": entry["flickr_url"], "sha256": IMAGES[entry["id"]]})
    return paths, images, sources
