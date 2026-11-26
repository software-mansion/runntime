"""The text every text reference runs on: questions from SQuAD and the
passages they are about."""

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

# SQuAD v1.1 dev set (Rajpurkar et al., 2016), CC BY-SA 4.0.
REPO = "rajpurkar/squad"
REVISION = "7b6d24c440a36b6815f21b70d25016731768db1f"
FILE = "plain_text/validation-00000-of-00001.parquet"
# A long passage is this many paragraphs of its article joined.
LONG_PARAGRAPHS = 5

ATTRIBUTION = {
    "dataset": "SQuAD v1.1, validation split",
    "authors": "Rajpurkar et al., 2016",
    "repo": REPO,
    "revision": REVISION,
    "license": "CC BY-SA 4.0",
}


def squad_pairs(pairs: int, long: int) -> tuple[list[str], list[str]]:
    """The first question of each of the first `pairs` articles and the
    passage it is about: the article's first paragraph, or its first
    paragraphs joined for the first `long` of them."""
    path = hf_hub_download(REPO, FILE, repo_type="dataset", revision=REVISION)
    rows = pq.read_table(path, columns=["title", "context", "question"]).to_pylist()
    # One row per question, in article and paragraph order. Each article
    # keeps its first question and its paragraphs, each once.
    questions: dict[str, str] = {}
    paragraphs: dict[str, dict[str, None]] = {}
    for row in rows:
        questions.setdefault(row["title"], row["question"])
        paragraphs.setdefault(row["title"], {}).setdefault(row["context"])
    titles = list(questions)[:pairs]
    passages = [
        "\n".join(list(paragraphs[title])[: LONG_PARAGRAPHS if i < long else 1]) for i, title in enumerate(titles)
    ]
    return [questions[title] for title in titles], passages
