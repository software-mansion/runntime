# runntime-tools

Quantizes a safetensors checkpoint to runntime's int8/int4 weight format, so
a model downloads smaller and runs on the quantized matmul.

```bash
uv run runntime quantize model.safetensors model.int8.safetensors --bits 8
```

- The source is a `.safetensors` file or a Hugging Face model folder, sharded
  or not. The output is one `.safetensors` file.
- 2D tensors named `*.weight` are quantized in groups of `--group-size`
  inputs (default 64), each group with one f16 scale. Embeddings and
  `lm_head` stay as stored. `--include` and `--exclude` take glob patterns
  and replace those defaults.
- Every other tensor is copied as stored, and the header metadata is kept.
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
