"""CLI for L2CS gaze calibration tool."""

import argparse
import sys
from pathlib import Path

from calibration_tool.calibration import CalibrationSession


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="L2CS Gaze Detection Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage from L2CS-Net root directory:
  .venv/bin/python -m calibration_tool
  .venv/bin/python -m calibration_tool --device cpu
  .venv/bin/python -m calibration_tool --windowed
  .venv/bin/python -m calibration_tool --camera 1

Examples:
  .venv/bin/python -m calibration_tool --model models/L2CSNet_gaze360.pkl
  .venv/bin/python -m calibration_tool --device gpu:0 --camera 0
        """,
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to L2CS model weights (default: models/L2CSNet_gaze360.pkl)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="gpu:0",
        help="Device for inference: 'gpu:0', 'gpu:1', 'cpu' (default: gpu:0)",
    )

    parser.add_argument("--camera", type=int, default=0, help="Webcam device ID (default: 0)")

    parser.add_argument(
        "--windowed", action="store_true", help="Use windowed mode instead of fullscreen"
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=9,
        choices=[9, 25],
        help="Calibration grid size: 9 (3×3) or 25 (5×5) points (default: 9)",
    )

    args = parser.parse_args()

    # Run calibration session
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
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
