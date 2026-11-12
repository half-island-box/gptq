#!/usr/bin/env python3
"""Run FP16, RTN, GPTQ, and AWQ Llama experiments sequentially."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run a Llama model through four experiments: FP16, RTN, GPTQ, and AWQ."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="llama-7b",
        help="Llama-7B alias/ID, or a local checkpoint path for compatibility (default: %(default)s)",
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="c4",
        choices=["wikitext2", "ptb", "c4"],
        help="calibration dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--wbits", type=int, default=4, choices=(2, 3, 4, 8),
        help="bits used by the RTN and GPTQ runs (default: %(default)s)",
    )
    parser.add_argument(
        "--results-csv", default="results.csv",
        help="shared CSV file for all experiment results (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the four commands without running them",
    )
    args, extra_args = parser.parse_known_args()
    if args.model in {"wikitext2", "ptb", "c4"} and args.dataset == "c4":
        if extra_args and not extra_args[0].startswith("-"):
            parser.error("pass both MODEL and DATASET, or use the default Llama-7B model")
        args.dataset = args.model
        args.model = parser.get_default("model")
    script = Path(__file__).resolve().parent / "llama.py"
    results_csv = str(Path(args.results_csv).expanduser().resolve())
    common_args = [
        str(script), args.model, args.dataset,
        "--results-csv", results_csv,
        *extra_args,
    ]

    experiments = [
        ("FP16", ["--wbits", "16"]),
        ("RTN", ["--wbits", str(args.wbits), "--nearest"]),
        ("GPTQ", ["--wbits", str(args.wbits), "--quant_method", "gptq"]),
        ("AWQ", ["--wbits", str(args.wbits), "--quant_method", "awq"]),
    ]

    print(f"Results CSV: {Path(results_csv)}")
    for method, method_args in experiments:
        command = [sys.executable, *common_args, *method_args]
        print(f"\n=== Starting {method} experiment ===")
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=script.parent)
        if result.returncode:
            print(f"{method} experiment failed with exit code {result.returncode}.", file=sys.stderr)
            raise SystemExit(result.returncode)

    if args.dry_run:
        print("\nDry run complete; no experiments were executed.")
    else:
        print("\nAll Llama experiments completed successfully.")


if __name__ == "__main__":
    main()
