"""The only public command surface for the CoWear release."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from . import DATASET_URL, PROTOCOL_VERSION, STORAGE_ROLE_FOR_PAPER, __version__
from .protocol.config import load_config
from .protocol.data import validate_paper_split
from .protocol.metrics import summarize_ates


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split-index", type=Path, default=None)
    parser.add_argument("--weights-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/run"))
    parser.add_argument("--device", default="cuda")


def _resolve_split(args: argparse.Namespace) -> Path:
    if args.split_index is not None:
        return args.split_index
    if args.data_root is not None:
        candidate = args.data_root / "session_split_seed2027.csv"
        if candidate.is_file():
            return candidate
        candidate = args.data_root / "splits" / "session_split_seed2027.csv"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "--split-index is required; download the split index from the CoWear dataset "
        f"release at {DATASET_URL}"
    )


def validate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    split = _resolve_split(args)
    counts = validate_paper_split(split)
    payload = {
        "package_version": __version__,
        "protocol_version": config.protocol_version,
        "dataset_version": config.dataset_version,
        "dataset_url": DATASET_URL,
        "split_index": str(split),
        "split_counts": counts,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def report(args: argparse.Namespace) -> None:
    """Normalize an existing per-session ATE CSV into the paper summary."""
    source = args.source
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "ate_m" not in rows[0]:
        raise ValueError(f"expected a non-empty CSV with an ate_m column: {source}")
    summary = summarize_ates([float(row["ate_m"]) for row in rows])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"source": str(source), "metric": "median per-session horizontal ATE", **summary}
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def train(args: argparse.Namespace) -> None:
    """Run the paper CoWear LSTM trainer with explicit public paths."""
    if args.data_root is None or args.split_index is None or args.output_dir is None:
        raise ValueError("train requires --data-root, --split-index and --output-dir")
    from .models import cowear_lstm

    target_mode = "self" if args.target_mode == "self" else STORAGE_ROLE_FOR_PAPER[args.target_mode]
    forwarded = [
        "--processed-root", str(args.data_root),
        "--split-index", str(args.split_index),
        "--cache-dir", str(args.output_dir / "cache"),
        "--output-dir", str(args.output_dir / "checkpoints"),
        "--target-mode", target_mode,
        "--input-frame", "device",
        "--calibration-mode", "published_extrinsic",
        "--device", args.device,
        "--seed", str(args.seed),
    ]
    if args.split_watch_hand:
        forwarded.append("--split-watch-hand")
    old_argv = sys.argv
    try:
        sys.argv = ["cowear train", *forwarded]
        cowear_lstm.main()
    finally:
        sys.argv = old_argv


def train_baseline(args: argparse.Namespace) -> None:
    """Dispatch one of the four published external baseline trainers."""
    from .baselines import get_baseline

    if args.method == "cowear_lstm":
        return train(args)
    if args.data_root is None or args.split_index is None:
        raise ValueError("train-baseline requires --data-root and --split-index")
    spec = get_baseline(args.method)
    forwarded = [
        "all",
        "--processed-root", str(args.data_root),
        "--split-index", str(args.split_index),
        "--output-root", str(args.output_dir),
    ]
    if args.method != "pdr":
        forwarded.extend(["--role", args.role, "--eval-split", "test"])
    if args.device == "cpu" and args.method in {"ronin", "tlio"}:
        forwarded.append("--cpu")
    old_argv = sys.argv
    try:
        sys.argv = [f"cowear train-baseline {args.method}", *forwarded]
        spec.main()()
    finally:
        sys.argv = old_argv


def evaluate(args: argparse.Namespace) -> None:
    """Run the frozen paper evaluation from a release weights directory."""
    if args.data_root is None or args.split_index is None or args.weights_root is None:
        raise ValueError("evaluate requires --data-root, --split-index and --weights-root")
    load_config(args.config)
    validate_paper_split(args.split_index)
    from .evaluation import paper

    root = args.weights_root
    forwarded = [
        "--processed-root", str(args.data_root),
        "--split-index", str(args.split_index),
        "--benchmark-root", str(root / "task1_external_baselines"),
        "--self-checkpoint-dir", str(root / "task1_cowear_lstm_self_checkpoints"),
        "--common-mobile-dir", str(root / "task2_phone_target_checkpoints"),
        "--common-watch-dir", str(root / "task2_watch_target_checkpoints"),
        "--common-rokid-dir", str(root / "task2_glasses_target_checkpoints"),
        "--cache-dir", str(args.output_dir / "cache"),
        "--output-dir", str(args.output_dir),
        "--device", args.device,
    ]
    old_argv = sys.argv
    try:
        sys.argv = ["cowear evaluate", *forwarded]
        paper.main()
    finally:
        sys.argv = old_argv


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m cowear")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate the public split and protocol")
    _common(validate_parser)
    validate_parser.set_defaults(handler=validate)

    report_parser = subparsers.add_parser("report", help="summarize a per-session ATE CSV")
    _common(report_parser)
    report_parser.add_argument("--source", type=Path, required=True)
    report_parser.set_defaults(handler=report)

    train_parser = subparsers.add_parser("train", help="train the paper CoWear LSTM")
    _common(train_parser)
    train_parser.add_argument("--target-mode", choices=("self", "phone", "watch", "glasses"), default="phone")
    train_parser.add_argument("--seed", type=int, default=2027)
    train_parser.add_argument("--split-watch-hand", action="store_true")
    train_parser.set_defaults(handler=train)

    baseline_parser = subparsers.add_parser(
        "train-baseline", help="train/evaluate PDR, RIDI, RoNIN, or TLIO"
    )
    _common(baseline_parser)
    baseline_parser.add_argument("--method", choices=("pdr", "ridi", "ronin", "tlio", "cowear_lstm"), required=True)
    baseline_parser.add_argument("--role", choices=("mobile", "watch", "rokid"), default="mobile")
    baseline_parser.add_argument("--target-mode", choices=("self", "phone", "watch", "glasses"), default="self")
    baseline_parser.add_argument("--seed", type=int, default=2027)
    baseline_parser.add_argument("--split-watch-hand", action="store_true")
    baseline_parser.set_defaults(handler=train_baseline)

    evaluate_parser = subparsers.add_parser("evaluate", help="reproduce all paper tables from release weights")
    _common(evaluate_parser)
    evaluate_parser.set_defaults(handler=evaluate)

    args = parser.parse_args(argv)
    args.handler(args)
