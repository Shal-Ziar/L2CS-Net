"""Diagnostic script to inspect calibration data structure."""

import numpy as np
from calibration_tool.calculate_calibration import load_calibration_data
from pathlib import Path

calib_file = Path("calibration_tool/calibration_data/calibration_2026-06-14_12-44-52.jsonl")
calib = load_calibration_data(calib_file)

print("=== CALIBRATION DATA DIAGNOSTIC ===\n")
print(f"Screen: {calib.screen_width}×{calib.screen_height}")
print(f"Number of calibration points: {len(calib.calibration_points)}\n")

print("Calibration Points Grid:")
print("-" * 80)
print(
    f"{'ID':<3} {'Pixel X':<8} {'Pixel Y':<8} {'N Readings':<12} {'Yaw (avg)':<12} {'Pitch (avg)':<12}"
)
print("-" * 80)

for i, point in enumerate(calib.calibration_points):
    n_readings = len(point.GazeResults)
    pixel_x, pixel_y = point.pixel_coordinates

    yaw_vals = np.array(
        [
            result.yaw.item() if hasattr(result.yaw, "item") else float(result.yaw)
            for result in point.GazeResults
        ]
    )
    pitch_vals = np.array(
        [
            result.pitch.item() if hasattr(result.pitch, "item") else float(result.pitch)
            for result in point.GazeResults
        ]
    )

    avg_yaw = yaw_vals.mean()
    avg_pitch = pitch_vals.mean()

    print(
        f"{i + 1:<3} {pixel_x:<8} {pixel_y:<8} {n_readings:<12} {avg_yaw:<12.4f} {avg_pitch:<12.4f}"
    )

print("\n" + "=" * 80)
print("Expected 3×3 Grid Layout:")
print("  (Xmin, Ymin) -------- (Xmid, Ymin) -------- (Xmax, Ymin)")
print("       |                    |                    |")
print("  (Xmin, Ymid) -------- (Xmid, Ymid) -------- (Xmax, Ymid)")
print("       |                    |                    |")
print("  (Xmin, Ymax) -------- (Xmid, Ymax) -------- (Xmax, Ymax)")
