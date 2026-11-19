"""Moonshine reference: the encoder states, the greedy tokens, the logits of
every teacher-forced step and the text from transformers, in f32.

    uv run python -m runntime_parity.moonshine tiny
    uv run python -m runntime_parity.moonshine base
"""

import gc
import sys

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoProcessor, MoonshineForConditionalGeneration

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


def repo(size: str) -> str:
    return f"UsefulSensors/moonshine-{size}"


def load(size: str, double: bool = False) -> MoonshineForConditionalGeneration:
    """The model on the CPU: in f32, or in f64 with `double`."""
    model = MoonshineForConditionalGeneration.from_pretrained(
        repo(size), revision=REVISIONS[size], dtype=torch.float32
    ).eval()
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


def noises(reference: Outputs, double: Outputs) -> dict[str, float]:
    """The reference noise of each metric: the largest difference between
    the f32 and the f64 outputs over the clips."""
    pairs = list(zip(reference, double, strict=True))
    tops = [np.argsort(-logits, axis=1)[:, :2] for _, logits in reference]
    return {
        "encoderError": max(noise(a, b) for (a, _), (b, _) in pairs),
        "encoderCosineDistance": max(cosine_noise(a.reshape(1, -1), b.reshape(1, -1)) for (a, _), (b, _) in pairs),
        "logitsError": max(noise(a, b) for (_, a), (_, b) in pairs),
        "logitsCosineDistance": max(cosine_noise(a, b) for (_, a), (_, b) in pairs),
        "logitGap": max(noise(gap(a, top), gap(b, top)) for ((_, a), (_, b)), top in zip(pairs, tops, strict=True)),
    }


def main() -> None:
    size = sys.argv[1]
    clip_ids, clips, transcripts = librispeech.librispeech_clips(CLIPS, MAX_SECONDS)
    processor = AutoProcessor.from_pretrained(repo(size), revision=REVISIONS[size])
    rate = librispeech.SAMPLE_RATE
    inputs = [processor(clip, sampling_rate=rate, return_tensors="pt") for clip in clips]

    model = load(size)
    tokens = []
    for clip_id, clip, features in zip(clip_ids, clips, inputs, strict=True):
        with torch.inference_mode():
            generated = model.generate(**features, max_length=int(len(clip) * TOKENS_PER_SECOND / rate))[0]
        # runntime's token cap is no lower: a clip that ends here ends there too.
        if generated[-1] != model.config.eos_token_id:
            raise RuntimeError(f"{clip_id} reaches the token cap")
        tokens.append(generated)
    reference = teacher_forced(model, inputs, tokens)
    for clip_id, ids, (_, logits) in zip(clip_ids, tokens, reference, strict=True):
        if not np.array_equal(logits.argmax(axis=1), ids[1:].numpy()):
            raise RuntimeError(f"{clip_id}: teacher forcing picks other tokens than the greedy run")
    texts = [processor.decode(ids, skip_special_tokens=True) for ids in tokens]
    # The same model in f64 gives the reference noise. transformers keeps its
    # RoPE in f32, so the noise is f32 against mostly f64.
    del model
    gc.collect()
    double = teacher_forced(load(size, double=True), inputs, tokens)

    tensors: dict[str, NDArray] = {}
    for i, (clip, ids, (states, logits)) in enumerate(zip(clips, tokens, reference, strict=True)):
        tensors[f"samples.{i}"] = clip
        tensors[f"tokens.{i}"] = ids.numpy().astype(np.int32)
        tensors[f"encoder.{i}"] = states
        tensors[f"logits.{i}"] = logits
    save(
        f"moonshine-{size}",
        {"clips": clip_ids, "transcripts": transcripts},
        tensors,
        {
            "repo": repo(size),
            "revision": REVISIONS[size],
            "dtype": "f32",
            "inputs": librispeech.ATTRIBUTION | {"count": len(clips)},
            "settings": {"maxSeconds": MAX_SECONDS, "tokensPerSecond": TOKENS_PER_SECOND},
            "noise": noises(reference, double),
        },
        ["torch", "transformers"],
        outputs={"texts": texts},
    )


if __name__ == "__main__":
    main()
