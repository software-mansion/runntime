"""Moonshine reference: the encoder states, the greedy tokens, the logits of
every teacher-forced step and the text from transformers, in f32.

    uv run python -m runntime_parity.moonshine tiny
    uv run python -m runntime_parity.moonshine base
"""

import gc
import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoProcessor, MoonshineForConditionalGeneration, PreTrainedModel

from . import librispeech
from .common import cosine_noise, noise, save

REVISIONS = {
    "tiny": "390624ed33d594443aa4aa221f5b9f283b545b5a",
    "base": "7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf",
}

# One clip of each of the first CLIPS speakers.
CLIPS = 12
# runntime's chunk length.
MAX_SECONDS = 26
# The most tokens a second of audio gives, from the model card.
TOKENS_PER_SECOND = 6.5

Outputs = list[tuple[NDArray[np.floating], NDArray[np.floating]]]
TeacherForced = Callable[[PreTrainedModel, list[dict[str, torch.Tensor]], list[torch.Tensor]], Outputs]


@dataclass(frozen=True)
class Reference:
    """A released model and the family its reference is saved as."""

    family: str
    repo: str
    revision: str
    model: type[PreTrainedModel]
    # runntime's chunk length: it splits a longer clip.
    max_seconds: int


def load(reference: Reference, double: bool = False) -> PreTrainedModel:
    """The model on the CPU: in f32, or in f64 with `double`."""
    model = reference.model.from_pretrained(reference.repo, revision=reference.revision, dtype=torch.float32).eval()
    return model.double() if double else model


def teacher_forced(
    model: MoonshineForConditionalGeneration, inputs: list[dict[str, torch.Tensor]], tokens: list[torch.Tensor]
) -> Outputs:
    """The encoder states and the logits of every step of each clip, the
    decoder fed that clip's `tokens`."""
    dtype = next(model.parameters()).dtype
    outputs = []
    with torch.inference_mode():
        for clip, ids in zip(inputs, tokens, strict=True):
            out = model(
                input_values=clip["input_values"].to(dtype),
                attention_mask=clip["attention_mask"],
                decoder_input_ids=ids[None, :-1],
            )
            outputs.append((out.encoder_last_hidden_state[0].numpy(), out.logits[0].numpy()))
    return outputs


def gap(logits: NDArray[np.floating], top: NDArray[np.intp]) -> NDArray[np.floating]:
    """How far the logit of the first token in each row of `top` is above
    the second's, per step."""
    steps = np.arange(len(logits))
    return logits[steps, top[:, 0]] - logits[steps, top[:, 1]]


def noises(outputs: Outputs, double: Outputs, reference: Outputs) -> dict[str, float]:
    """The noise of each metric: the largest difference between `outputs`
    and the f64 outputs over the clips. The logit gap is between the two
    tokens `reference` scores highest."""
    pairs = list(zip(outputs, double, strict=True))
    tops = [np.argsort(-logits, axis=1)[:, :2] for _, logits in reference]
    return {
        "encoderError": max(noise(a, b) for (a, _), (b, _) in pairs),
        "encoderCosineDistance": max(cosine_noise(a.reshape(1, -1), b.reshape(1, -1)) for (a, _), (b, _) in pairs),
        "logitsError": max(noise(a, b) for (_, a), (_, b) in pairs),
        "logitsCosineDistance": max(cosine_noise(a, b) for (_, a), (_, b) in pairs),
        "logitGap": max(noise(gap(a, top), gap(b, top)) for ((_, a), (_, b)), top in zip(pairs, tops, strict=True)),
    }


def save_reference(reference: Reference, teacher_forced: TeacherForced) -> None:
    """Runs the model of `reference` on the speech and saves its outputs:
    `teacher_forced` gives the encoder states and the logits of a model."""
    clip_ids, clips, transcripts = librispeech.librispeech_clips(CLIPS, reference.max_seconds)
    processor = AutoProcessor.from_pretrained(reference.repo, revision=reference.revision)
    rate = librispeech.SAMPLE_RATE
    inputs = [processor(clip, sampling_rate=rate, return_tensors="pt") for clip in clips]

    model = load(reference)
    tokens = []
    for clip_id, clip, features in zip(clip_ids, clips, inputs, strict=True):
        with torch.inference_mode():
            generated = model.generate(**features, max_length=int(len(clip) * TOKENS_PER_SECOND / rate))[0]
        # runntime's token cap is no lower: a clip that ends here ends there too.
        if generated[-1] != model.config.eos_token_id:
            raise RuntimeError(f"{clip_id} reaches the token cap")
        tokens.append(generated)
    outputs = teacher_forced(model, inputs, tokens)
    for clip_id, ids, (_, logits) in zip(clip_ids, tokens, outputs, strict=True):
        if not np.array_equal(logits.argmax(axis=1), ids[1:].numpy()):
            raise RuntimeError(f"{clip_id}: teacher forcing picks other tokens than the greedy run")
    texts = [processor.decode(ids, skip_special_tokens=True) for ids in tokens]
    # The same model in f64 gives the reference noise. transformers keeps its
    # RoPE in f32, so the noise is f32 against mostly f64.
    del model
    gc.collect()
    double = teacher_forced(load(reference, double=True), inputs, tokens)
    # The same model in f16 gives the f16 noise, its outputs as f32.
    gc.collect()
    half = [
        (a.astype(np.float32), b.astype(np.float32)) for a, b in teacher_forced(load(reference).half(), inputs, tokens)
    ]

    tensors: dict[str, NDArray] = {}
    for i, (clip, ids, (states, logits)) in enumerate(zip(clips, tokens, outputs, strict=True)):
        tensors[f"samples.{i}"] = clip
        tensors[f"tokens.{i}"] = ids.numpy().astype(np.int32)
        tensors[f"encoder.{i}"] = states
        tensors[f"logits.{i}"] = logits
    save(
        reference.family,
        {"clips": clip_ids, "transcripts": transcripts},
        tensors,
        {
            "repo": reference.repo,
            "revision": reference.revision,
            "dtype": "f32",
            "inputs": librispeech.ATTRIBUTION | {"count": len(clips)},
            "settings": {"maxSeconds": reference.max_seconds, "tokensPerSecond": TOKENS_PER_SECOND},
            "noise": noises(outputs, double, outputs),
            "noiseF16": noises(half, double, outputs),
        },
        ["torch", "transformers"],
        outputs={"texts": texts},
    )


def main() -> None:
    size = sys.argv[1]
    save_reference(
        Reference(
            family=f"moonshine-{size}",
            repo=f"UsefulSensors/moonshine-{size}",
            revision=REVISIONS[size],
            model=MoonshineForConditionalGeneration,
            max_seconds=MAX_SECONDS,
        ),
        teacher_forced,
    )


if __name__ == "__main__":
    main()
