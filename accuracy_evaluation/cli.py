"""CLI for the static-point gaze evaluation toolkit: `mock`, `live`, `aggregate`."""

from __future__ import annotations

import argparse
import sys
import uuid
from accuracy_evaluation.evaluate import (
    CONDITIONS,
    aggregate_trials,
    compute_trial_metrics,
    generate_static_points,
    run_live_trial,
    run_mock_trial,
)
from pathlib import Path


def _add_common_trial_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--session-id", default=None, help="Default: random 8-char id")
    parser.add_argument("--trial-id", default="t01")
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--condition", default="no_calibration", choices=CONDITIONS)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--dwell", type=float, default=1.5, help="Seconds recorded per point")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out-dir", default="accuracy_evaluation_output")
    parser.add_argument("--threshold-px", type=float, default=20.0)


def _out_path(args: argparse.Namespace) -> Path:
    args.session_id = args.session_id or str(uuid.uuid4())[:8]
    return (
        Path(args.out_dir)
        / f"{args.session_id}_{args.trial_id}_{args.condition}_rep{args.rep}.jsonl"
    )


def cmd_mock(args: argparse.Namespace) -> int:
    screen = (args.screen_w, args.screen_h)
    points = generate_static_points(*screen, rows=args.rows, cols=args.cols)
    out_path = _out_path(args)

    print(f"[mock] writing trial to {out_path}")
    run_mock_trial(
        args.session_id,
        args.trial_id,
        args.rep,
        args.condition,
        points,
        out_path,
        fps=args.fps,
        dwell=args.dwell,
        screen_size=screen,
        seed=args.seed,
    )
    print("Metrics:", compute_trial_metrics(out_path, threshold_px=args.threshold_px))
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    from calibration_tool.calibration import GazeCapture

    out_path = _out_path(args)

    print("[live] loading model and camera (this can take a few seconds)...")
    gaze_capture = GazeCapture(model_path=args.model, device=args.device, camera_id=args.camera)
    try:
        n_points = args.rows * args.cols
        print(f"[live] recording {n_points} points, {args.dwell}s each. Press ESC to abort.")
        result_path = run_live_trial(
            args.session_id,
            args.trial_id,
            args.rep,
            args.condition,
            out_path,
            gaze_capture,
            fit_type=args.fit_type,
            calibration_path=args.calibration,
            ridge_alpha=args.ridge_alpha,
            fov_rad=args.fov_rad,
            rows=args.rows,
            cols=args.cols,
            dwell=args.dwell,
            fps=args.fps,
            fullscreen=not args.windowed,
            windowed_size=(args.screen_w, args.screen_h),
            model_version=args.model_version,
        )
    finally:
        gaze_capture.cleanup()

    if result_path is None:
        print("[live] trial aborted before any frames were recorded; nothing saved.")
        return 1

    print(f"[live] wrote {result_path}")
    print("Metrics:", compute_trial_metrics(result_path, threshold_px=args.threshold_px))
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    paths = sorted(Path(args.out_dir).glob(args.pattern))
    if not paths:
        print(f"No files matched {args.pattern!r} in {args.out_dir}")
        return 1
    print(f"Aggregating {len(paths)} trial(s): {[p.name for p in paths]}")
    print(aggregate_trials(paths, threshold_px=args.threshold_px))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Static-point gaze evaluation toolkit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mock_parser = subparsers.add_parser("mock", help="Simulate a trial (no camera/model needed)")
    _add_common_trial_args(mock_parser)
    mock_parser.add_argument("--screen-w", type=int, default=1280)
    mock_parser.add_argument("--screen-h", type=int, default=720)
    mock_parser.add_argument(
        "--seed", type=int, default=None, help="RNG seed for the mock predictor"
    )
    mock_parser.set_defaults(func=cmd_mock)

    live_parser = subparsers.add_parser("live", help="Run a real trial with camera + L2CS model")
    _add_common_trial_args(live_parser)
    live_parser.add_argument(
        "--model", type=Path, default=Path("models/l2cs_gaze360_resnet50.safetensors")
    )
    live_parser.add_argument("--device", default="cpu", help="'cpu', 'gpu:0', etc.")
    live_parser.add_argument("--camera", type=int, default=0)
    live_parser.add_argument(
        "--fit-type",
        choices=("none", "univariate", "bivariate"),
        default="none",
        help="'none' uses a rough uncalibrated linear estimate",
    )
    live_parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Calibration JSONL, required for univariate/bivariate --fit-type",
    )
    live_parser.add_argument("--ridge-alpha", type=float, default=1.0)
    live_parser.add_argument(
        "--fov-rad",
        type=float,
        default=0.35,
        help="Assumed +/- gaze range spanning the screen when --fit-type=none",
    )
    live_parser.add_argument(
        "--windowed", action="store_true", help="Run in a window, not fullscreen"
    )
    live_parser.add_argument("--screen-w", type=int, default=1920, help="Window size if --windowed")
    live_parser.add_argument("--screen-h", type=int, default=1080, help="Window size if --windowed")
    live_parser.add_argument("--model-version", default="l2cs")
    live_parser.set_defaults(func=cmd_live)

    agg_parser = subparsers.add_parser(
        "aggregate", help="Aggregate metrics across repeated trial logs"
    )
    agg_parser.add_argument(
        "pattern", help="Glob pattern relative to --out-dir, e.g. '*_rep*.jsonl'"
    )
    agg_parser.add_argument("--out-dir", default="accuracy_evaluation_output")
    agg_parser.add_argument("--threshold-px", type=float, default=20.0)
    agg_parser.set_defaults(func=cmd_aggregate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
