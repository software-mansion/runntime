# runntime-tools

Quantizes a safetensors checkpoint to runntime's int8/int4 weight format, so
a model downloads smaller and runs on the quantized matmul.

The tool knows the file format, not the model: it never asks which layer
reads a tensor or what a kernel can run. runntime fits each weight to its
layer at load, and loads a weight no quantized kernel runs as float.

```bash
uv run runntime quantize model.safetensors model.int8.safetensors --bits 8
```

- The source is a `.safetensors` file or a Hugging Face model folder, sharded
  or not. The output is one `.safetensors` file.
- Float tensors named `*.weight` with 2 or more dims are quantized, under
  their own names. `--include GLOB` replaces `*.weight`.
- Tensors matching `*embed*`, `*pos_emb*` or `lm_head.*` stay as stored. This
  is a quality default, not a format rule: quantized embedding tables only
  rarely hold up. `--exclude GLOB` replaces it, so `--exclude none` quantizes
  them too.
- A quantized tensor is stored as rows, `[first dim, everything else]`: a
  Conv1d weight `[out, in, k]` becomes `[out, in·k]`. Each row is cut into
  groups of `--group-size` values (default 32, any positive integer), each
  group with one f16 scale, stored as `<name>_scale`.
- A selected tensor stays as stored only when the format cannot hold it: it
  is not float, it has fewer than 2 dims, the group size does not divide its
  row, or, for int4, its row is odd (two values pack into a byte).
- The header's `runntime.quant` entry is `{"version":1,"scheme":"symmetric"}`.
  The reader takes each tensor's bits from its dtype (`I8` int8, `U8` int4)
  and its group size from the shape of its scales.
- Every other tensor is copied as stored, or as F16 with `--f16`, and the
  header metadata is kept.
- `--drop` takes glob patterns of tensors to leave out, like a pooler the
  model never reads.
- It prints the max and mean absolute error of every quantized tensor and
  why each other tensor was skipped. Exclude the layers that lose too much.

## Development

```bash
uv sync
uv run pytest
uv run ruff check && uv run ruff format --check
uv run pyright
```

`tests/golden` holds a small float checkpoint and its int4 and int8 versions.
The TypeScript tests in `packages/runntime` read the same files. After a
change to the output, regenerate them with
`uv run runntime quantize tests/golden/tiny.safetensors tests/golden/tiny.int8.safetensors --bits 8 --group-size 32`
(and the same with `--bits 4`).
