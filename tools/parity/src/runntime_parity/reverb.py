"""Reverb reference: what the sampler of reverb-tts sees in every pass of a
take, its noise, the codes it ends with and the codec's audio, in f32. One
reference per release variant, each run on that variant's weights.

    uv run python -m runntime_parity.reverb fp32
    uv run python -m runntime_parity.reverb fp16
    uv run python -m runntime_parity.reverb int8
    uv run python -m runntime_parity.reverb int4
"""

import gc
import sys

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from reverb_tts import tts as package
from reverb_tts.model import MASK
from reverb_tts.sampler import Batch, sample
from reverb_tts.text import prepare_text
from reverb_tts.tts import CODEC_REVISION, TTS, Voice

from .common import noise, save

REPO = "software-mansion/Reverb-TTS"
# The release tag v1.0.0.
REVISION = "72be799c8fb6d56a206d260b425c48bf3bd5943d"
SEED = 0

# A preset voice of the release and a sentence for it, a short and a long
# one per language, with numbers and units for the text preparation.
INPUTS = [
    ("en_male_narrator_1", "en", "The train leaves at 7:45."),
    (
        "en_female_narrator",
        "en",
        "In 2023 the bridge was closed for 14 weeks, and the repairs cost $2.5 million, "
        "about 30% more than the city had planned.",
    ),
    ("de_female_standard", "de", "Das Paket wiegt 3,5 kg."),
    (
        "de_male_standard",
        "de",
        "Am 12. Mai 2021 fuhren wir 480 km nach Hamburg, und die Fahrt dauerte wegen 2 Baustellen fast 6 Stunden.",
    ),
    ("fr_female_standard", "fr", "Il fait 18 °C ce matin."),
    (
        "fr_male_standard",
        "fr",
        "Le musée ouvre à 9 h 30 et accueille près de 1 200 visiteurs par jour, soit 15 % de plus qu'en 2019.",
    ),
    ("es_male_standard_1", "es", "La reunión empieza a las 10:15."),
    (
        "es_male_standard_2",
        "es",
        "El río mide 910 km de largo y, en 2022, su caudal bajó un 40 % tras 5 meses sin lluvia.",
    ),
    ("it_male_standard", "it", "Il biglietto costa 12 euro."),
    (
        "it_male_expressive",
        "it",
        "La torre è alta 85 m, fu costruita nel 1372 e ogni anno la visitano circa 250.000 persone.",
    ),
    ("pl_male_standard", "pl", "Autobus odjeżdża o 6:20."),
    (
        "pl_male_standard",
        "pl",
        "W 2020 roku trasa miała 37 km, a dziś liczy już 52 km i codziennie korzysta z niej ponad 8 tys. rowerzystów.",
    ),
    ("pt_female_standard", "pt", "A loja fecha às 19h."),
    (
        "pt_male_standard",
        "pt",
        "A ponte tem 2,3 km de comprimento, foi inaugurada em 1998 e custou cerca de 900 milhões de euros.",
    ),
]

Stats = dict[str, NDArray]


class Recorded(Batch):
    """A batch of one take that keeps its prompt and, per pass, the grid it
    was given, the codebooks asked for and the log-probabilities."""

    made: list["Recorded"] = []  # noqa: RUF012

    def __init__(self, network: torch.nn.Module, prompts: list, polish_bias: float) -> None:
        super().__init__(network, prompts, polish_bias)
        self.prompts, self.passes, self.codes = prompts, [], None
        Recorded.made.append(self)

    def log_probs(self, tokens: torch.Tensor, codebooks: range, polish: bool = False) -> torch.Tensor:
        out = super().log_probs(tokens, codebooks, polish)
        self.passes.append((tokens[0].clone(), codebooks, polish, out[0]))
        return out


def recorded_sample(batch: Recorded, seeds: list[int], config: object) -> list[torch.Tensor]:
    codes = sample(batch, seeds, config)
    batch.codes = codes[0]
    return codes


def load(variant: str, double: bool = False) -> TTS:
    """The release variant on the CPU: in f32, or in f64 with `double`."""
    dtype = torch.float64 if double else torch.float32
    return TTS.from_pretrained(REPO, variant=variant, revision=REVISION, device="cpu", dtype=dtype)


def slots(
    lp: torch.Tensor, score: torch.Tensor, top: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per slot of a pass: the two best codes by `score` (or `top`, given),
    the log-probability of the first and how far its score is above the
    second's."""
    top = score.topk(2, -1).indices if top is None else top
    picked = score.gather(-1, top)
    return top, lp.gather(-1, top[..., :1])[..., 0], picked[..., 0] - picked[..., 1]


def measured(passes: list, guidance: float, tops: list[torch.Tensor] | None = None) -> tuple[Stats, list[torch.Tensor]]:
    """What the passes of one take give per slot, as the sampler reads them:
    the 8 main passes over codebooks 0-1, with the log-probability of the
    code each slot holds, then the first polish pass over codebooks 2-7 and
    the second, guided against the first, over 4-7. `tops` are the best
    codes of another run of the same grids, to measure this one at."""
    main, (_, _, _, first), (_, _, _, second) = passes[:-2], passes[-2], passes[-1]
    scores = [lp for _, _, _, lp in main] + [first, second + guidance * (second - first[2:])]
    lps = [lp for _, _, _, lp in main] + [first, second]
    found = [
        slots(lp, score, None if tops is None else tops[i])
        for i, (lp, score) in enumerate(zip(lps, scores, strict=True))
    ]
    held = [lp.gather(-1, grid[:2].clamp(max=MASK - 1)[..., None])[..., 0] for grid, _, _, lp in main]

    def stack(index: int, of: slice) -> NDArray:
        return torch.stack([f[index] for f in found[of]]).numpy()

    steps = slice(0, len(main))
    stats = {
        "coarseBest": stack(0, steps)[..., 0].astype(np.int32),
        "coarseConfidence": stack(1, steps),
        "coarseGap": stack(2, steps),
        "coarseHeld": torch.stack(held).numpy(),
        "fineBest": found[-2][0][..., 0].numpy().astype(np.int32),
        "fineConfidence": found[-2][1].numpy(),
        "fineGap": found[-2][2].numpy(),
        "guidedBest": found[-1][0][..., 0].numpy().astype(np.int32),
        "guidedConfidence": found[-1][1].numpy(),
        "guidedGap": found[-1][2].numpy(),
    }
    return stats, [f[0] for f in found]


def noises(reference: list[Stats], double: list[Stats], grids: list[NDArray]) -> dict[str, float]:
    """The reference noise of each metric: the largest difference between
    the f32 and the f64 run over the takes. The held code's counts only
    where a slot holds one."""
    pairs = list(zip(reference, double, strict=True))
    confidences = ("coarseConfidence", "fineConfidence", "guidedConfidence")
    gaps = ("coarseGap", "fineGap", "guidedGap")
    holds = [grid[: len(a["coarseHeld"]), :2] != MASK for grid, a in zip(grids, reference, strict=True)]
    return {
        "confidenceError": max(noise(a[name], b[name]) for a, b in pairs for name in confidences),
        "heldConfidenceError": max(
            noise(a["coarseHeld"][held], b["coarseHeld"][held]) for (a, b), held in zip(pairs, holds, strict=True)
        ),
        "logProbGap": max(noise(a[name], b[name]) for a, b in pairs for name in gaps),
    }


def main() -> None:
    variant = sys.argv[1]
    package.Batch, package.sample = Recorded, recorded_sample
    voices = [Voice.load(hf_hub_download(REPO, f"voices/{name}.json", revision=REVISION)) for name, _, _ in INPUTS]

    tts = load(variant)
    guidance = tts.sampler.polish_guidance
    takes, reference, tops, tensors, inputs = [], [], [], {}, []
    for i, ((name, language, text), voice) in enumerate(zip(INPUTS, voices, strict=True)):
        tts.generate(text, voice, language, seed=SEED, trim=False)
        take = Recorded.made.pop()
        if Recorded.made or len(take.prompts) != 1:
            raise RuntimeError(f"{text!r} is spoken in pieces")
        stats, top = measured(take.passes, guidance)
        frames = take.frames[0]
        # The noise the sampler draws: [2, frames] per main pass, from the take's seed.
        generator = torch.Generator().manual_seed(SEED)
        drawn = torch.stack([torch.rand((2, frames), generator=generator) for _ in range(tts.sampler.steps)])
        with torch.inference_mode():
            audio = tts.codec.decode(take.codes[None]).audio_values[0, 0].float()
        grids = torch.stack([grid for grid, _, _, _ in take.passes]).numpy().astype(np.int32)
        takes.append((take.prompts, [(grid, codebooks, polish) for grid, codebooks, polish, _ in take.passes]))
        reference.append(stats)
        tops.append(top)
        tensors |= {f"{key}.{i}": value for key, value in stats.items()}
        tensors |= {
            f"voice.{i}": voice.codes.numpy().astype(np.int32),
            f"grids.{i}": grids,
            f"noise.{i}": drawn.numpy(),
            f"codes.{i}": take.codes.numpy().astype(np.int32),
            f"audio.{i}": audio.numpy(),
        }
        inputs.append(
            {
                "voice": name,
                "language": language,
                "text": text,
                "prepared": prepare_text(text, language),
                "transcript": voice.text,
                "speechFrames": voice.speech_frames,
                "frames": frames,
            }
        )
    polish_bias = tts.sampler.polish_bias
    # The same weights in f64, on the same grids, give the reference noise.
    # reverb-tts casts the head's output to f32 before the log-softmax, and
    # transformers keeps its RoPE in f32, so the noise is f32 against mostly
    # f64.
    del tts
    gc.collect()
    network = load(variant, double=True).network
    double = []
    with torch.inference_mode():
        for (prompts, passes), top in zip(takes, tops, strict=True):
            batch = Batch(network, prompts, polish_bias)
            again = [
                (grid, codebooks, polish, batch.log_probs(grid[None], codebooks, polish)[0])
                for grid, codebooks, polish in passes
            ]
            double.append(measured(again, guidance, top)[0])

    grids = [tensors[f"grids.{i}"] for i in range(len(INPUTS))]
    save(
        f"reverb-{variant}",
        inputs,
        tensors,
        {
            "repo": REPO,
            "revision": REVISION,
            "dtype": "f32",
            "inputs": {
                "voices": f"{REPO} at {REVISION}, voices/",
                "license": "CC BY-NC 4.0",
                "count": len(INPUTS),
            },
            "settings": {"variant": variant, "seed": SEED, "codec": CODEC_REVISION},
            "noise": noises(reference, double, grids),
        },
        ["torch", "transformers", "reverb-tts", "verbalis"],
    )


if __name__ == "__main__":
    main()
