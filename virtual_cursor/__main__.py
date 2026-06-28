"""CLI entry point for virtual cursor application."""

import argparse
from pathlib import Path
from virtual_cursor.cursor_app import CursorApp
from virtual_cursor.smoother import MovingAverageSmoother
from virtual_cursor.trial_waypoint import WaypointTrial


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Virtual cursor gaze tracking application")
    parser.add_argument(
        "--calibration",
        type=str,
        required=True,
        help="Path to calibration JSONL file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to gaze model weights (uses default if not specified)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Torch device",
    )
    parser.add_argument(
        "--cam",
        type=int,
        default=0,
        help="Camera device ID",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        default=False,
        help="Display in fullscreen mode",
    )
    parser.add_argument(
        "--trial",
        action="store_true",
        help="Run waypoint trial (4-corner square) instead of just cursor",
    )
    parser.add_argument(
        "--dwell-time",
        type=float,
        default=2.0,
        help="Dwell time in seconds for trial targets (default: 2.0)",
    )
    parser.add_argument(
        "--fit-type",
        type=str,
        default="univariate",
        choices=["univariate", "bivariate"],
        help="Calibration fit type: univariate (independent pitch→x, yaw→y) or bivariate (with pitch×yaw cross-terms)",
    )
    parser.add_argument(
        "--averaging-window",
        type=int,
        default=20,
        help="Number of frames to average cursor position over (1 = no smoothing, default: 1)",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength for bivariate polynomial fit (default: 1.0, ignored for univariate)",
    )
    args = parser.parse_args()

    # Validate calibration file
    calib_path = Path(args.calibration)
    if not calib_path.exists():
        print(f"Error: Calibration file not found: {calib_path}")
        return

    # Initialize cursor app
    try:
        smoother = (
            MovingAverageSmoother(args.averaging_window) if args.averaging_window > 1 else None
        )
        app = CursorApp(
            calibration_path=calib_path,
            model_path=args.model,
            device=args.device,
            camera_id=args.cam,
            fullscreen=args.fullscreen,
            fit_type=args.fit_type,
            smoother=smoother,
            ridge_alpha=args.ridge_alpha,
        )
    except Exception as e:
        print(f"Error initializing app: {e}")
        return

    # Run cursor or trial
    try:
        if args.trial:
            trial = WaypointTrial(app, dwell_threshold=args.dwell_time)
            trial.run_trial()
        else:
            app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()
