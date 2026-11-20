"""Moonshine streaming reference: the encoder states the decoder reads, the
greedy tokens, the logits of every teacher-forced step and the text from
transformers, in f32.

    uv run python -m runntime_parity.moonshine_streaming
"""

import torch
from transformers import MoonshineStreamingForConditionalGeneration

from .moonshine import Outputs, Reference, save_reference

REFERENCE = Reference(
    family="moonshine-streaming-tiny",
    repo="UsefulSensors/moonshine-streaming-tiny",
    revision="f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a",
    model=MoonshineStreamingForConditionalGeneration,
    max_seconds=30,
)


def teacher_forced(
    model: MoonshineStreamingForConditionalGeneration,
    inputs: list[dict[str, torch.Tensor]],
    tokens: list[torch.Tensor],
) -> Outputs:
    """The encoder states as the decoder reads them, with the positions
    added, and the logits of every step of each clip, the decoder fed that
    clip's `tokens`."""
    dtype = next(model.parameters()).dtype
    decoder = model.model.decoder
    outputs = []
    with torch.inference_mode():
        for clip, ids in zip(inputs, tokens, strict=True):
            # Without the mask the encoder attends outside its windows.
            encoded = model.model.encoder(clip["input_values"].to(dtype), attention_mask=clip["attention_mask"])
            states = encoded.last_hidden_state[0]
            # Taken before the model call, which adds the positions into `encoded` in place.
            adapted = decoder.proj(states + decoder.pos_emb(torch.arange(len(states))))
            logits = model(encoder_outputs=encoded, decoder_input_ids=ids[None, :-1]).logits[0]
            outputs.append((adapted.numpy(), logits.numpy()))
    return outputs


def main() -> None:
    save_reference(REFERENCE, teacher_forced)


if __name__ == "__main__":
    main()
