"""all-MiniLM-L6-v2 reference: unit-length embeddings of questions and
passages from sentence-transformers, in f32.

    uv run python -m runntime_parity.minilm
"""

import numpy as np
import torch
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from . import squad
from .common import cosine_noise, noise, save

REPO = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

# One question and its paragraph from each of the first PAIRS articles.
PAIRS = 25
# The first LONG passages are joined paragraphs, longer than the model reads.
LONG = 3
# Passages ranked per question, best first.
TOP_K = 5


def truncated(texts: list[str]) -> int:
    """How many of `texts` are longer than the model reads."""
    model = SentenceTransformer(REPO, revision=REVISION, device="cpu")
    lengths = [len(ids) for ids in model.tokenizer(texts)["input_ids"]]
    return sum(length > model.max_seq_length for length in lengths)


def embed(texts: list[str], dtype: torch.dtype = torch.float32) -> NDArray[np.floating]:
    """The embeddings of `texts` on the CPU, computed in `dtype`. f16 ones
    come back as f32."""
    model = SentenceTransformer(REPO, revision=REVISION, device="cpu").to(dtype)
    embeddings = model.encode(texts, batch_size=1, normalize_embeddings=True, convert_to_tensor=True)
    return (embeddings.float() if dtype == torch.float16 else embeddings).numpy()


def main() -> None:
    queries, passages = squad.squad_pairs(PAIRS, LONG)
    if truncated(passages[:LONG]) != LONG:
        raise RuntimeError(f"the first {LONG} passages have to be longer than the model reads")
    reference = {"queries": embed(queries), "passages": embed(passages)}
    scores = reference["queries"] @ reference["passages"].T
    ranking = np.argsort(-scores, axis=1, kind="stable")[:, :TOP_K].astype(np.int32)
    # The same model in f64 gives the reference noise, and in f16 the f16 noise.
    double = {"queries": embed(queries, torch.float64), "passages": embed(passages, torch.float64)}
    half = {"queries": embed(queries, torch.float16), "passages": embed(passages, torch.float16)}
    save(
        "minilm",
        {"queries": queries, "passages": passages},
        reference | {"ranking": ranking},
        {
            "repo": REPO,
            "revision": REVISION,
            "dtype": "f32",
            "inputs": squad.ATTRIBUTION | {"count": len(queries) + len(passages)},
            "settings": {"topK": TOP_K},
            "noise": {
                "error": max(noise(reference[name], double[name]) for name in reference),
                "cosineDistance": max(cosine_noise(reference[name], double[name]) for name in reference),
                "score": noise(scores, double["queries"] @ double["passages"].T),
            },
            "noiseF16": {
                "error": max(noise(half[name], double[name]) for name in half),
                "cosineDistance": max(cosine_noise(half[name], double[name]) for name in half),
                "score": noise(half["queries"] @ half["passages"].T, double["queries"] @ double["passages"].T),
            },
        },
        ["torch", "sentence-transformers", "transformers"],
    )


if __name__ == "__main__":
    main()
