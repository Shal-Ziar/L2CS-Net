"""CLI for personal_trainer.

Commands
--------
collect
    Run the calibration UI, capture face crops with face-detector-only
    (no gaze model), and write them to a session directory.

finetune
    Generate pseudo-labels with an existing model, then fine-tune it on
    the collected crops and save a dated .safetensors file.

run
    ``collect`` followed immediately by ``finetune`` — the full pipeline
    in one command.

Examples
--------
::

    # Step 1: collect face crops
    python -m personal_trainer collect --windowed

    # Step 2: fine-tune on collected session
    python -m personal_trainer finetune \\
        --snapshot models/l2cs_gaze360_resnet50.safetensors \\
        --data-dir training_data/<session-id>

    # Or both steps in one go
    python -m personal_trainer run \\
        --snapshot models/l2cs_gaze360_resnet50.safetensors
"""

import argparse
from pathlib import Path

from personal_trainer.collect import ImageCollectionSession
from personal_trainer.finetune import generate_pseudo_labels, run_finetune


# ---------------------------------------------------------------------------
# Shared argument helpers
# ---------------------------------------------------------------------------


def _add_gpu_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        metavar="ID",
        help='GPU device id or "cpu" (default: 0)',
    )


def _add_collect_args(parser: argparse.ArgumentParser, include_gpu: bool = True) -> None:
    parser.add_argument(
        "--camera", type=int, default=0, metavar="ID", help="Webcam device id (default: 0)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_data"),
        metavar="DIR",
        help="Root directory for session output (default: training_data/)",
    )
    parser.add_argument(
        "--windowed", action="store_true", help="Use windowed mode instead of fullscreen"
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=9,
        choices=[4, 9, 25],
        metavar="{4,9,25}",
        help="Number of calibration points (default: 9)",
    )
    if include_gpu:
        _add_gpu_arg(parser)


def _add_finetune_args(
    parser: argparse.ArgumentParser, include_gpu: bool = True
) -> None:  # noqa: E501
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        metavar="PATH",
        help="Base model weights (.pkl or .safetensors)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="ResNet50",
        metavar="ARCH",
        help="Model architecture matching snapshot (default: ResNet50)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="Fine-tuning epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="Mini-batch size (default: 8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="Learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        metavar="A",
        help="MSE regression loss weight (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/personal"),
        metavar="DIR",
        help="Directory for fine-tuned model (default: output/personal/)",
    )
    if include_gpu:
        _add_gpu_arg(parser)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _cmd_collect(args: argparse.Namespace) -> Path:
    session = ImageCollectionSession(
        output_dir=args.output_dir,
        camera_id=args.camera,
        gpu=args.gpu,
        fullscreen=not args.windowed,
        grid_size=args.grid_size,
    )
    return session.run()


def _cmd_finetune(args: argparse.Namespace, data_dir: Path) -> Path:
    generate_pseudo_labels(
        data_dir=data_dir,
        snapshot=args.snapshot,
        arch=args.arch,
        gpu=args.gpu,
        batch_size=32,
    )
    return run_finetune(
        data_dir=data_dir,
        snapshot=args.snapshot,
        arch=args.arch,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        output_dir=args.output,
        gpu=args.gpu,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="personal_trainer",
        description="Personalised gaze fine-tuning: collect face crops then fine-tune L2CS.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- collect -------------------------------------------------------------
    collect_parser = subparsers.add_parser(
        "collect",
        help="Capture face crops during calibration (no gaze model required).",
    )
    _add_collect_args(collect_parser)

    # -- finetune ------------------------------------------------------------
    finetune_parser = subparsers.add_parser(
        "finetune",
        help="Generate pseudo-labels and fine-tune model on a collected session.",
    )
    finetune_parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Session directory produced by the collect step.",
    )
    _add_finetune_args(finetune_parser)

    # -- run (collect + finetune) -------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Full pipeline: collect face crops then fine-tune in one step.",
    )
    _add_collect_args(run_parser, include_gpu=False)
    _add_finetune_args(run_parser, include_gpu=False)
    _add_gpu_arg(run_parser)  # shared by both collect and finetune steps

    args = parser.parse_args()

    if args.command == "collect":
        _cmd_collect(args)

    elif args.command == "finetune":
        _cmd_finetune(args, args.data_dir)

    elif args.command == "run":
        session_dir = _cmd_collect(args)
        _cmd_finetune(args, session_dir)
