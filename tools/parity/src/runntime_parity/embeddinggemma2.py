"""EmbeddingGemma 2 reference: unit-length embeddings of text, images and
audio from sentence-transformers, in f32.

    uv run python -m runntime_parity.embeddinggemma2
"""

import gc

import numpy as np
import torch
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from . import coco, librispeech, squad
from .common import cosine_noise, noise, save

REPO = "google/embeddinggemma-2"
REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"

# Pairs of each modality.
PAIRS = 8
# The first LONG passages are joined paragraphs, longer than LONG_TOKENS, the
# text model's sliding window.
LONG = 2
LONG_TOKENS = 512
# The most tokens of an input and seconds of a clip runntime reads.
MAX_TOKENS = 2048
MAX_SECONDS = 30
# Items ranked per asking text, best first.
TOP_K = 5

# The inputs each tower embeds: their vectors share one noise.
TOWERS = {
    "text": ("queries", "passages", "captions", "transcripts"),
    "image": ("images",),
    "audio": ("clips",),
}
# The asking texts of each modality and what they rank.
MODALITIES = {
    "text": ("queries", "passages"),
    "image": ("captions", "images"),
    "audio": ("transcripts", "clips"),
}
# The prompt of each text input. Images and audio take none.
PROMPTS = {"queries": "query", "passages": "document", "captions": "query", "transcripts": "query"}

Embeddings = dict[str, NDArray[np.floating]]


def load(double: bool = False) -> SentenceTransformer:
    """The model on the CPU: in f32, or in f64 with `double`."""
    model = SentenceTransformer(REPO, revision=REVISION, device="cpu", model_kwargs={"dtype": torch.float32})
    if double:
        model = model.double()
        # The reference casts audio attention to f32, so its scale stays f32.
        for name, parameter in model[0].auto_model.audio_tower.named_parameters():
            if name.endswith("per_dim_scale"):
                parameter.data = parameter.data.float()
    return model


def tokens(model: SentenceTransformer, name: str, text: str) -> int:
    """How many tokens `text` takes with the prompt of `name`."""
    return model.preprocess([model.prompts[PROMPTS[name]] + text])["input_ids"].shape[1]


def embed(model: SentenceTransformer, inputs: dict[str, list]) -> Embeddings:
    """The embeddings of every list in `inputs`, one input at a time."""
    return {
        name: model.encode(
            items,
            prompt_name=PROMPTS.get(name),
            batch_size=1,
            normalize_embeddings=True,
            convert_to_tensor=True,
        ).numpy()
        for name, items in inputs.items()
    }


def scores(embeddings: Embeddings, modality: str) -> NDArray[np.floating]:
    """The score of every ranked item, for each asking text of `modality`."""
    asking, ranked = MODALITIES[modality]
    return embeddings[asking] @ embeddings[ranked].T


def noises(embeddings: Embeddings, double: Embeddings) -> dict[str, float]:
    """The noise of each metric: how far `embeddings` are from the f64 ones,
    per tower."""
    out = {}
    for modality, names in TOWERS.items():
        out[f"{modality}Error"] = max(noise(embeddings[name], double[name]) for name in names)
        out[f"{modality}CosineDistance"] = max(cosine_noise(embeddings[name], double[name]) for name in names)
        out[f"{modality}Score"] = noise(scores(embeddings, modality), scores(double, modality))
    return out


def main() -> None:
    queries, passages = squad.squad_pairs(PAIRS, LONG)
    paths, images, sources = coco.coco_images(PAIRS)
    captions = coco.coco_captions([source["id"] for source in sources])
    ids, clips, transcripts = librispeech.librispeech_clips(PAIRS)
    if max(len(clip) for clip in clips) > MAX_SECONDS * librispeech.SAMPLE_RATE:
        raise RuntimeError(f"every clip has to be at most {MAX_SECONDS} s")
    inputs = {
        "queries": queries,
        "passages": passages,
        "captions": captions,
        "images": images,
        "transcripts": transcripts,
        "clips": [{"array": clip, "sampling_rate": librispeech.SAMPLE_RATE} for clip in clips],
    }

    model = load()
    lengths = {name: [tokens(model, name, text) for text in inputs[name]] for name in PROMPTS}
    if min(lengths["passages"][:LONG]) <= LONG_TOKENS:
        raise RuntimeError(f"the first {LONG} passages have to be longer than {LONG_TOKENS} tokens")
    if max(max(counts) for counts in lengths.values()) > MAX_TOKENS:
        raise RuntimeError(f"every text has to be at most {MAX_TOKENS} tokens")
    reference = embed(model, inputs)
    # The same model in f64 gives the reference noise. The f32 one goes first.
    # transformers keeps the norms, softmax and RoPE in f32 there, so the noise
    # is f32 against mostly f64.
    del model
    gc.collect()
    double = embed(load(double=True), inputs)
    # The same model in f16 gives the f16 noise, its embeddings as f32.
    gc.collect()
    half = {name: vectors.astype(np.float32) for name, vectors in embed(load().half(), inputs).items()}

    rankings = {
        f"{modality}Ranking": np.argsort(-scores(reference, modality), axis=1, kind="stable")[:, :TOP_K].astype(
            np.int32
        )
        for modality in MODALITIES
    }
    save(
        "embeddinggemma2",
        {
            "queries": queries,
            "passages": passages,
            "captions": captions,
            "images": paths,
            "transcripts": transcripts,
            "clips": ids,
        },
        reference | rankings | {f"samples.{i}": clip for i, clip in enumerate(clips)},
        {
            "repo": REPO,
            "revision": REVISION,
            "dtype": "f32",
            "inputs": {
                "license": "CC BY-SA 4.0, CC BY 2.0, CC BY 4.0",
                "count": sum(len(items) for items in inputs.values()),
                "text": squad.ATTRIBUTION,
                "images": coco.ATTRIBUTION | {"images": sources},
                "audio": librispeech.ATTRIBUTION,
            },
            "settings": {"topK": TOP_K},
            "noise": noises(reference, double),
            "noiseF16": noises(half, double),
        },
        ["torch", "sentence-transformers", "transformers", "pillow", "soundfile"],
    )


if __name__ == "__main__":
    main()
