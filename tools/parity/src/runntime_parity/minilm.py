"""all-MiniLM-L6-v2 reference: unit-length embeddings of questions and
passages from sentence-transformers, in f32.

    uv run python -m runntime_parity.minilm
"""

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from .common import cosine_noise, devices, noise, save

REPO = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

# SQuAD v1.1 dev set (Rajpurkar et al., 2016), CC BY-SA 4.0.
SQUAD_REPO = "rajpurkar/squad"
SQUAD_REVISION = "7b6d24c440a36b6815f21b70d25016731768db1f"
SQUAD_FILE = "plain_text/validation-00000-of-00001.parquet"
# One question and its paragraph from each of the first PAIRS articles.
PAIRS = 25
# The first LONG passages are the first LONG_PARAGRAPHS paragraphs of their
# article joined, longer than the model reads.
LONG = 3
LONG_PARAGRAPHS = 5
# Passages ranked per question, best first.
TOP_K = 5


def squad_pairs() -> tuple[list[str], list[str]]:
    """The first question of each article and the passage it is about: the
    article's first paragraph, or its first paragraphs joined for a long one."""
    path = hf_hub_download(SQUAD_REPO, SQUAD_FILE, repo_type="dataset", revision=SQUAD_REVISION)
    rows = pq.read_table(path, columns=["title", "context", "question"]).to_pylist()
    # One row per question, in article and paragraph order. Each article
    # keeps its first question and its paragraphs, each once.
    questions: dict[str, str] = {}
    paragraphs: dict[str, dict[str, None]] = {}
    for row in rows:
        questions.setdefault(row["title"], row["question"])
        paragraphs.setdefault(row["title"], {}).setdefault(row["context"])
    titles = list(questions)[:PAIRS]
    passages = [
        "\n".join(list(paragraphs[title])[: LONG_PARAGRAPHS if i < LONG else 1]) for i, title in enumerate(titles)
    ]
    return [questions[title] for title in titles], passages


def truncated(texts: list[str]) -> int:
    """How many of `texts` are longer than the model reads."""
    model = SentenceTransformer(REPO, revision=REVISION, device="cpu")
    lengths = [len(ids) for ids in model.tokenizer(texts)["input_ids"]]
    return sum(length > model.max_seq_length for length in lengths)


def embed(texts: list[str], device: str) -> NDArray[np.float32]:
    model = SentenceTransformer(REPO, revision=REVISION, device=device)
    return model.encode(texts, batch_size=1, normalize_embeddings=True, convert_to_numpy=True)


def main() -> None:
    queries, passages = squad_pairs()
    if truncated(passages[:LONG]) != LONG:
        raise RuntimeError(f"the first {LONG} passages have to be longer than the model reads")
    device, noise_device = devices()
    reference = {"queries": embed(queries, device), "passages": embed(passages, device)}
    scores = reference["queries"] @ reference["passages"].T
    ranking = np.argsort(-scores, axis=1, kind="stable")[:, :TOP_K].astype(np.int32)
    second = {"queries": embed(queries, noise_device), "passages": embed(passages, noise_device)}
    save(
        "minilm",
        {"queries": queries, "passages": passages},
        reference | {"ranking": ranking},
        {
            "repo": REPO,
            "revision": REVISION,
            "dtype": "f32",
            "inputs": {
                "dataset": "SQuAD v1.1, validation split",
                "authors": "Rajpurkar et al., 2016",
                "repo": SQUAD_REPO,
                "revision": SQUAD_REVISION,
                "license": "CC BY-SA 4.0",
            },
            "devices": [device, noise_device],
            "noise": {
                "error": max(noise(reference[name], second[name]) for name in reference),
                "cosineDistance": max(cosine_noise(reference[name], second[name]) for name in reference),
            },
        },
        ["torch", "sentence-transformers", "transformers"],
    )


if __name__ == "__main__":
    main()
