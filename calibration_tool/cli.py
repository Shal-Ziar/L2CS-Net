"""CLI for L2CS gaze calibration tool."""

import argparse
import sys
from calibration_tool.calibration import CalibrationSession
from pathlib import Path


def cmd_calibrate(args):
    """Run full 9-point calibration session."""
    try:
        print("=" * 60)
        print("L2CS Gaze Detection Calibration Tool")
        print("=" * 60)
        print()
        print("Instructions:")
        print("  1. Look at each point in order (1-9)")
        print("  2. Press SPACEBAR to select a point (turns green)")
        print("  3. Press SPACEBAR again to record for 2 seconds (turns blue)")
        print("  4. Point turns grey when recording is complete")
        print("  5. Press ESC to abort calibration at any time")
        print()

        session = CalibrationSession(
            model_path=args.model,
            device=args.device,
            camera_id=args.camera,
            fullscreen=not args.windowed,
            grid_size=args.grid_size,
        )

        output_path = session.run()

        if output_path:
            print()
            print("=" * 60)
            print(f"Calibration saved: {output_path}")
            print("Visualizations generated in same directory.")
            print("=" * 60)
            return 0
        else:
            print()
            print("Calibration aborted or incomplete.")
            return 1

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def cmd_daemon(args):
    """Run background micro-calibration daemon."""
    try:
        from calibration_tool.cursor_calibration import CursorCalibrationDaemon

        print("=" * 60)
        print("L2CS Background Micro-Calibration Daemon")
        print("=" * 60)
        print()
        print("Daemon running in background.")
        print("Press 'c' key anytime to trigger 5-point calibration checkpoint.")
        print("Results appended to most recent calibration file.")
        print("Press Ctrl+C to exit.")
        print()

        daemon = CursorCalibrationDaemon(
            model_path=args.model,
            device=args.device,
            camera_id=args.camera,
            calib_file=args.calib_file,
        )

        daemon.run()
        return 0

    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Ensure pynput is installed: pip install pynput", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="L2CS Gaze Detection Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Calibrate command
    calib_parser = subparsers.add_parser("calibrate", help="Run full calibration session")
    calib_parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to L2CS model weights (default: models/L2CSNet_gaze360.pkl)",
    )
    calib_parser.add_argument(
        "--device",
        type=str,
        default="gpu:0",
        help="Device for inference: 'gpu:0', 'gpu:1', 'cpu' (default: gpu:0)",
    )
    calib_parser.add_argument("--camera", type=int, default=0, help="Webcam device ID (default: 0)")
    calib_parser.add_argument(
        "--windowed", action="store_true", help="Use windowed mode instead of fullscreen"
    )
    calib_parser.add_argument(
        "--grid-size",
        type=int,
        default=9,
        choices=[9, 25],
        help="Calibration grid size: 9 (3×3) or 25 (5×5) points (default: 9)",
    )
    calib_parser.set_defaults(func=cmd_calibrate)

    # Daemon command
    daemon_parser = subparsers.add_parser("daemon", help="Run background micro-calibration daemon")
    daemon_parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to L2CS model weights (default: models/L2CSNet_gaze360.pkl)",
    )
    daemon_parser.add_argument(
        "--device",
        type=str,
        default="gpu:0",
        help="Device for inference: 'gpu:0', 'gpu:1', 'cpu' (default: gpu:0)",
    )
    daemon_parser.add_argument(
        "--camera", type=int, default=0, help="Webcam device ID (default: 0)"
    )
    daemon_parser.add_argument(
        "--calib-file",
        type=Path,
        default=None,
        help="Calibration file to append to (default: most recent)",
    )
    daemon_parser.set_defaults(func=cmd_daemon)

    args = parser.parse_args()

    # If no command specified, default to calibrate (backward compatibility)
    if not hasattr(args, "func"):
        # Default to calibrate command for backward compatibility
        args.model = None
        args.device = "gpu:0"
        args.camera = 0
        args.windowed = False
        args.grid_size = 9
        return cmd_calibrate(args)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
