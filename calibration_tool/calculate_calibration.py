import numpy as np
import numpy.typing as npt
from calibration_tool.types import Calibration
from pathlib import Path

DEFAULT_PATH = Path(__file__).parent.joinpath(
    "calibration_data/calibration_2026-06-14_12-44-52.jsonl"
)


def load_calibration_data(file_path: Path = DEFAULT_PATH) -> Calibration:
    with open(file_path, "r") as f:
        calibration_data = Calibration.model_validate_json(f.read())
    return calibration_data


def calculate_polynomial_mapping(calibration: Calibration) -> npt.NDArray[np.float64]:
    # restructure data such that x,y are compared to
    # coordinates x, y (width, height)
    # split all calibration points into x and y coordinates and shape them as (pixel_x, yaw) and (pixel_y, pitch)
    # turn them into a numpy array
    x_array = np.array([], dtype=np.float64).reshape(0, 2)
    y_array = np.array([], dtype=np.float64).reshape(0, 2)

    for point in calibration.calibration_points:
        num_results = len(point.GazeResults)

        # x-> yaw mapping
        coord_array_x = np.zeros((num_results, 2), dtype=np.float64)
        coord_array_x[:, 0] = float(point.pixel_coordinates[0])
        # Extract scalar values from numpy arrays
        coord_array_x[:, 1] = np.array(
            [
                result.yaw.item() if hasattr(result.yaw, "item") else float(result.yaw)
                for result in point.GazeResults
            ]
        )

        # y-> pitch mapping
        coord_array_y = np.zeros((num_results, 2), dtype=np.float64)
        coord_array_y[:, 0] = point.pixel_coordinates[1]
        # Extract scalar values from numpy arrays
        coord_array_y[:, 1] = np.array(
            [
                result.pitch.item() if hasattr(result.pitch, "item") else float(result.pitch)
                for result in point.GazeResults
            ]
        )

        # Concatenate arrays
        if x_array.size == 0:
            x_array = coord_array_x
            y_array = coord_array_y
        else:
            x_array = np.concatenate([x_array, coord_array_x], axis=0)
            y_array = np.concatenate([y_array, coord_array_y], axis=0)

    polyfit_y = np.polyfit(x_array[:, 0], x_array[:, 1], deg=3)
    polyfit_x = np.polyfit(y_array[:, 0], y_array[:, 1], deg=3)

    return np.array([polyfit_x, polyfit_y])


if __name__ == "__main__":
    calibration_data = load_calibration_data()
    calculate_polynomial_mapping(calibration_data)
