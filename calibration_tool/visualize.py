"""Post-calibration visualization for gaze calibration data."""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def parse_calibration_jsonl(jsonl_path: Path) -> Tuple[Dict, List[Dict]]:
    """
    Parse JSONL calibration file.

    Args:
        jsonl_path: Path to JSONL file

    Returns:
        Tuple of (metadata, calibration_points)
    """
    metadata = {}
    calibration_points = []

    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if i == 0 and data.get("type") == "calibration_metadata":
                metadata = data
            else:
                calibration_points.append(data)

    return metadata, calibration_points


def gaze_to_screen_coords(
    pitch: float,
    yaw: float,
    screen_width: int,
    screen_height: int,
    distance_mm: int = 600,  # Default distance in mm
) -> Tuple[float, float]:
    """
    Convert gaze angles (pitch, yaw) to screen coordinates.

    Args:
        pitch: Gaze pitch in radians
        yaw: Gaze yaw in radians
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
        distance_mm: Distance from eyes to screen in mm

    Returns:
        Tuple of (x, y) screen coordinates
    """
    # Convert angles to millimeters on the screen
    x_mm = distance_mm * math.tan(yaw)
    y_mm = distance_mm * math.tan(pitch)

    # Convert to screen center coordinates
    center_x = screen_width / 2
    center_y = screen_height / 2

    # Simple approximation: pixels per mm (assuming ~25 pixels per inch, 25.4mm per inch)
    pixels_per_mm = screen_width / 500  # Rough estimate

    screen_x = center_x + x_mm * pixels_per_mm
    screen_y = center_y + y_mm * pixels_per_mm

    # Clamp to screen bounds
    screen_x = max(0, min(screen_width, screen_x))
    screen_y = max(0, min(screen_height, screen_y))

    return screen_x, screen_y


def create_scatter_plot(metadata: Dict, calibration_points: List[Dict], output_path: Path) -> None:
    """
    Create a scatter plot of gaze points vs expected grid locations.

    Args:
        metadata: Calibration metadata
        calibration_points: List of calibration point data
        output_path: Path to save the plot
    """
    screen_width = metadata.get("screen_width", 1920)
    screen_height = metadata.get("screen_height", 1080)

    # Expected grid point locations (3x3)
    margin_x = int(screen_width * 0.1)
    margin_y = int(screen_height * 0.1)
    usable_width = screen_width - 2 * margin_x
    usable_height = screen_height - 2 * margin_y
    step_x = usable_width // 2
    step_y = usable_height // 2

    expected_points = {}
    point_id = 1
    for row in range(3):
        for col in range(3):
            x = margin_x + col * step_x
            y = margin_y + row * step_y
            expected_points[point_id] = (x, y)
            point_id += 1

    # Extract gaze points
    gaze_x = []
    gaze_y = []
    point_ids = []

    for cal_point in calibration_points:
        pt_id = cal_point["calibration_point"]
        gaze_results = cal_point["GazeResults"]

        for gaze_result in gaze_results:
            pitch = gaze_result["pitch"]
            yaw = gaze_result["yaw"]

            # Handle list or scalar values
            if isinstance(pitch, list):
                pitch = pitch[0] if pitch else 0
            if isinstance(yaw, list):
                yaw = yaw[0] if yaw else 0

            screen_x, screen_y = gaze_to_screen_coords(pitch, yaw, screen_width, screen_height)
            gaze_x.append(screen_x)
            gaze_y.append(screen_y)
            point_ids.append(pt_id)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot screen background
    ax.set_xlim(0, screen_width)
    ax.set_ylim(screen_height, 0)  # Invert Y for image coordinates
    ax.set_aspect("equal")

    # Plot expected grid points
    for pt_id, (exp_x, exp_y) in expected_points.items():
        ax.plot(exp_x, exp_y, "g^", markersize=12, label="Expected" if pt_id == 1 else "")
        ax.text(exp_x + 20, exp_y, str(pt_id), fontsize=10, color="green")

    # Plot captured gaze points
    scatter = ax.scatter(
        gaze_x, gaze_y, c=point_ids, cmap="tab10", s=20, alpha=0.6, label="Captured gaze"
    )

    # Styling
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title("Calibration Gaze Points: Expected vs Captured")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=ax, label="Calibration Point ID")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Scatter plot saved to {output_path}")


def create_heatmap(
    metadata: Dict, calibration_points: List[Dict], output_path: Path, bin_size: int = 50
) -> None:
    """
    Create a heatmap of gaze distribution across the screen.

    Args:
        metadata: Calibration metadata
        calibration_points: List of calibration point data
        output_path: Path to save the heatmap
        bin_size: Size of histogram bins in pixels
    """
    screen_width = metadata.get("screen_width", 1920)
    screen_height = metadata.get("screen_height", 1080)

    # Extract gaze points
    gaze_x = []
    gaze_y = []

    for cal_point in calibration_points:
        gaze_results = cal_point["GazeResults"]

        for gaze_result in gaze_results:
            pitch = gaze_result["pitch"]
            yaw = gaze_result["yaw"]

            # Handle list or scalar values
            if isinstance(pitch, list):
                pitch = pitch[0] if pitch else 0
            if isinstance(yaw, list):
                yaw = yaw[0] if yaw else 0

            screen_x, screen_y = gaze_to_screen_coords(pitch, yaw, screen_width, screen_height)
            gaze_x.append(screen_x)
            gaze_y.append(screen_y)

    # Create 2D histogram
    heatmap, xedges, yedges = np.histogram2d(
        gaze_x,
        gaze_y,
        bins=[screen_width // bin_size, screen_height // bin_size],
        range=[[0, screen_width], [0, screen_height]],
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot heatmap
    im = ax.imshow(
        heatmap.T,
        cmap="hot",
        origin="upper",
        extent=[0, screen_width, screen_height, 0],
        aspect="auto",
    )

    # Styling
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title("Gaze Distribution Heatmap")

    plt.colorbar(im, ax=ax, label="Gaze Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Heatmap saved to {output_path}")


def visualize_calibration(jsonl_path: Path, output_dir: Optional[Path] = None) -> None:
    """
    Generate scatter plot and heatmap visualizations from calibration data.

    Args:
        jsonl_path: Path to JSONL calibration file
        output_dir: Directory to save plots (defaults to same directory as JSONL)
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {jsonl_path}")

    # Set output directory
    if output_dir is None:
        output_dir = jsonl_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse data
    metadata, calibration_points = parse_calibration_jsonl(jsonl_path)

    if not calibration_points:
        print("No calibration points found in JSONL file.")
        return

    # Create plots
    session_id = jsonl_path.stem.replace("calibration_", "")

    scatter_path = output_dir / f"{session_id}_scatter.png"
    heatmap_path = output_dir / f"{session_id}_heatmap.png"

    create_scatter_plot(metadata, calibration_points, scatter_path)
    create_heatmap(metadata, calibration_points, heatmap_path)

    print(f"Visualizations complete: {scatter_path}, {heatmap_path}")
