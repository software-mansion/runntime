"""`runntime quantize`: writes a safetensors checkpoint in runntime's int8/int4
format v1 and prints the error of every quantized tensor."""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np

from runntime_tools.quantize import Bits, dequantize, pack_int4, quantize
from runntime_tools.safetensors_io import Tensor, from_array, read_checkpoint, to_float32, write_safetensors

QUANT_METADATA_KEY = "runntime.quant"
"""Header metadata key that marks a quantized file and holds its setting."""

FORMAT_VERSION = 1

FLOAT_DTYPES = frozenset({"F32", "F16", "BF16"})

DEFAULT_INCLUDE = ("*.weight",)

DEFAULT_EXCLUDE = ("*embed*", "lm_head.*")
"""runntime reads embeddings as float lookup tables, and lm_head often shares
the embedding's weights."""

IN_FEATURES_MULTIPLE = 8
"""runntime's quantized matmul reads 8 input rows per tile."""

PACKED_WORD_BITS = 32
"""runntime packs a weight 32 bits to a word, so out features must fill whole
words: a multiple of 4 for int8, 8 for int4."""


@dataclass(frozen=True)
class Options:
    bits: Bits
    group_size: int
    include: Sequence[str]
    exclude: Sequence[str]


def skip_reason(name: str, tensor: Tensor, options: Options) -> str | None:
    """Why `tensor` stays as stored, or None when it gets quantized."""
    if not any(fnmatchcase(name, pattern) for pattern in options.include):
        return "not matched by --include"
    excluded_by = next((pattern for pattern in options.exclude if fnmatchcase(name, pattern)), None)
    if excluded_by:
        return f"excluded by {excluded_by}"
    if tensor.dtype not in FLOAT_DTYPES:
        return f"dtype {tensor.dtype}"
    if len(tensor.shape) == 3:
        return "3D expert tensors are not supported yet"
    if len(tensor.shape) != 2:
        return f"{len(tensor.shape)}D, needs [out, in]"
    out_features, in_features = tensor.shape
    if in_features % IN_FEATURES_MULTIPLE or in_features % options.group_size:
        return f"in features {in_features} not a multiple of {IN_FEATURES_MULTIPLE} and of the group size"
    out_multiple = PACKED_WORD_BITS // options.bits
    if out_features % out_multiple:
        return f"out features {out_features} not a multiple of {out_multiple}"
    return None


def quantize_checkpoint(source: Path, destination: Path, options: Options) -> list[str]:
    """Quantizes the selected tensors of `source` into `destination` and
    returns one report line per tensor, in file order."""
    checkpoint = read_checkpoint(source)
    tensors: dict[str, Tensor] = {}
    report: list[str] = []
    quantized_count = 0
    for name, tensor in checkpoint.tensors.items():
        shape = f"[{', '.join(map(str, tensor.shape))}]"
        reason = skip_reason(name, tensor, options)
        if reason:
            tensors[name] = tensor
            report.append(f"skipped    {name} {shape}: {reason}")
            continue
        weight = to_float32(tensor)
        quantized = quantize(weight, options.bits, options.group_size)
        if options.bits == 8:
            tensors[name] = from_array("I8", quantized.ints)
        else:
            tensors[name] = from_array("U8", pack_int4(quantized.ints))
        tensors[f"{name}_scale"] = from_array("F16", quantized.scales)
        quantized_count += 1
        error = np.abs(dequantize(quantized) - weight)
        report.append(f"quantized  {name} {shape}: max error {error.max():.3g}, mean {error.mean():.3g}")
    if quantized_count == 0:
        raise ValueError("no tensor was quantized:\n" + "\n".join(report))
    setting = {"version": FORMAT_VERSION, "bits": options.bits, "groupSize": options.group_size, "scheme": "symmetric"}
    metadata = checkpoint.metadata | {QUANT_METADATA_KEY: json.dumps(setting, separators=(",", ":"))}
    write_safetensors(destination, tensors, metadata)
    return report


def group_size(value: str) -> int:
    """argparse type: a positive multiple of 4."""
    size = int(value)
    if size <= 0 or size % 4:
        raise argparse.ArgumentTypeError(f"{value} is not a positive multiple of 4")
    return size


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="runntime")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "quantize",
        help="quantize a safetensors checkpoint to int8 or int4",
        description="Quantizes the 2D weights of a safetensors checkpoint to runntime's int8/int4 format.",
    )
    command.add_argument("source", type=Path, help="a .safetensors file or a Hugging Face model folder")
    command.add_argument("destination", type=Path, help="the .safetensors file to write")
    command.add_argument("--bits", type=int, choices=(4, 8), required=True)
    command.add_argument("--group-size", type=group_size, default=64, help="inputs that share one scale (default: 64)")
    command.add_argument(
        "--include",
        action="append",
        metavar="GLOB",
        help=f"tensor names to quantize, repeatable, replaces the default {' '.join(DEFAULT_INCLUDE)}",
    )
    command.add_argument(
        "--exclude",
        action="append",
        metavar="GLOB",
        help=f"tensor names to keep as stored, repeatable, replaces the default {' '.join(DEFAULT_EXCLUDE)}",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    options = Options(
        bits=args.bits,
        group_size=args.group_size,
        include=args.include or DEFAULT_INCLUDE,
        exclude=args.exclude or DEFAULT_EXCLUDE,
    )
    try:
        report = quantize_checkpoint(args.source, args.destination, options)
    except ValueError as error:
        print(f"runntime quantize: {error}", file=sys.stderr)
        return 1
    print("\n".join(report))
    return 0
