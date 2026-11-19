"""The speech every audio reference runs on: LibriSpeech test-clean clips
with their transcripts, decoded once so the reference and the browser read
the same samples."""

import hashlib
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray

# LibriSpeech (Panayotov et al., 2015), CC BY 4.0.
REPO = "openslr/librispeech_asr"
REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
FILE = "clean/test/0000.parquet"
SHA256 = "7113aa4c3cf963fb54697145719a7725f984c8836d1c494a554cbb9f1a017df0"
SAMPLE_RATE = 16000

ATTRIBUTION = {
    "dataset": "LibriSpeech test-clean",
    "authors": "Panayotov et al., 2015",
    "repo": REPO,
    "revision": REVISION,
    "file": {"path": FILE, "sha256": SHA256},
    "license": "CC BY 4.0",
}


def librispeech_clips(
    count: int, max_seconds: float | None = None
) -> tuple[list[str], list[NDArray[np.float32]], list[str]]:
    """The first clip of each of the first `count` speakers, in file order:
    its id, its mono samples at 16 kHz, and its transcript. With
    `max_seconds`, a speaker's first clip that is no longer than that."""
    path = Path(hf_hub_download(REPO, FILE, repo_type="dataset", revision=REVISION))
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"{FILE} has sha256 {digest}, expected {SHA256}")
    first: dict[int, tuple[str, NDArray[np.float32], str]] = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
        for row in batch.to_pylist():
            if row["speaker_id"] in first:
                continue
            samples, rate = soundfile.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if rate != SAMPLE_RATE or samples.ndim != 1:
                raise RuntimeError(f"{row['id']} is not mono at {SAMPLE_RATE} Hz")
            if max_seconds is None or len(samples) <= max_seconds * SAMPLE_RATE:
                first[row["speaker_id"]] = (row["id"], samples, row["text"])
        if len(first) >= count:
            break
    chosen = list(first.values())[:count]
    if len(chosen) != count:
        raise RuntimeError(f"{FILE} has {len(chosen)} speakers, expected at least {count}")
    ids, clips, transcripts = zip(*chosen, strict=True)
    return list(ids), list(clips), list(transcripts)
