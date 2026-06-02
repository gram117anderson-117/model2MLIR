"""model2mlir command-line interface.

    model2mlir convert  <model.py> [--out out.mlir] [--output-type ...] [--quant SCHEME]
    model2mlir coverage <model.py>

`<model.py>` must expose `get_model_and_inputs() -> (nn.Module, tuple[Tensor, ...])`
(see examples/simple_mlp.py).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_model_module(path: str) -> Any:
    p = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(p.stem, p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load model module from {p}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_model_and_inputs"):
        raise RuntimeError(f"{p} must define get_model_and_inputs() -> (model, inputs)")
    return module.get_model_and_inputs()


def _make_quant(scheme: str | None) -> Any:
    if not scheme:
        return None
    from model2mlir.capture.torchao_pipeline import QuantizationConfig

    return QuantizationConfig(scheme=scheme)


def _cmd_convert(args: argparse.Namespace) -> int:
    import model2mlir

    model, inputs = _load_model_module(args.model)
    result = model2mlir.convert(
        model,
        tuple(inputs),
        output_type=args.output_type,
        quantization=_make_quant(args.quant),
    )
    if not result.ok:
        sys.stderr.write("conversion failed:\n  " + "\n  ".join(result.diagnostics) + "\n")
        return 1
    if args.out:
        Path(args.out).write_text(result.mlir_text)
        sys.stderr.write(f"[{result.path_taken}] wrote {len(result.mlir_text)} chars to {args.out}\n")
    else:
        sys.stdout.write(result.mlir_text)
    return 0


def _cmd_coverage(args: argparse.Namespace) -> int:
    import json

    import model2mlir

    model, inputs = _load_model_module(args.model)
    report = model2mlir.coverage_report(model, tuple(inputs), quantization=_make_quant(args.quant))
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="model2mlir", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("convert", help="convert a model to MLIR")
    pc.add_argument("model", help="path to a .py exposing get_model_and_inputs()")
    pc.add_argument("--out", help="output .mlir path (default: stdout)")
    pc.add_argument("--output-type", default="linalg-on-tensors")
    pc.add_argument("--quant", default=None, help="torchAO scheme name (e.g. int8_weight_only)")
    pc.set_defaults(func=_cmd_convert)

    pv = sub.add_parser("coverage", help="report op coverage for a model")
    pv.add_argument("model", help="path to a .py exposing get_model_and_inputs()")
    pv.add_argument("--quant", default=None)
    pv.set_defaults(func=_cmd_coverage)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
