"""Load calibration data and extract polynomial mappings."""

import numpy as np
from calibration_tool.calculate_calibration import (
    calculate_gaze_bounds,
    calculate_polynomial_mapping_bivariate,
    calculate_polynomial_mapping_univariate,
    load_calibration_data,
)
from pathlib import Path


class CalibrationLoader:
    """Load JSONL calibration file and extract screen dimensions + polynomials."""

    def __init__(
        self, calibration_path: Path | str, fit_type: str = "univariate", ridge_alpha: float = 1.0
    ):
        """Load calibration from file.

        Args:
            calibration_path: Path to calibration JSONL file
            fit_type: "univariate" (independent pitch→x, yaw→y) or "bivariate" (with cross-terms)
            ridge_alpha: Ridge regularization strength for bivariate fit (ignored for univariate)

        Raises:
            FileNotFoundError: If calibration file not found
            ValueError: If calibration data invalid
        """
        if fit_type not in ("univariate", "bivariate"):
            raise ValueError(f"Invalid fit_type: {fit_type}. Use 'univariate' or 'bivariate'.")

        self.calibration_path = Path(calibration_path)
        if not self.calibration_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {self.calibration_path}")

        # Load calibration data
        try:
            self.calibration = load_calibration_data(self.calibration_path)
        except Exception as e:
            raise ValueError(f"Failed to load calibration: {e}") from e

        # Extract screen dimensions
        self.screen_width: int = self.calibration.screen_width
        self.screen_height: int = self.calibration.screen_height
        self.fit_type = fit_type

        # Calculate polynomial mappings
        try:
            if fit_type == "univariate":
                poly_coefs = calculate_polynomial_mapping_univariate(self.calibration)
                self.poly_pitch_to_x = poly_coefs[0]  # pitch → pixel_x
                self.poly_yaw_to_y = poly_coefs[1]  # yaw → pixel_y
                self.coeffs_x = None
                self.coeffs_y = None
            else:  # bivariate
                self.coeffs_x, self.coeffs_y = calculate_polynomial_mapping_bivariate(
                    self.calibration, alpha=ridge_alpha
                )
                self.poly_pitch_to_x = None
                self.poly_yaw_to_y = None
        except Exception as e:
            raise ValueError(f"Failed to calculate polynomial mapping: {e}") from e

    def get_screen_dims(self) -> tuple[int, int]:
        """Get screen width and height.

        Returns:
            (width, height) tuple
        """
        return (self.screen_width, self.screen_height)

    def get_polynomials(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Get univariate polynomial coefficients (None if bivariate).

        Returns:
            (poly_pitch_to_x, poly_yaw_to_y) tuple
        """
        return (self.poly_pitch_to_x, self.poly_yaw_to_y)

    def get_bivariate_coeffs(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Get bivariate polynomial coefficients (None if univariate).

        Returns:
            (coeffs_x, coeffs_y) tuple
        """
        return (self.coeffs_x, self.coeffs_y)

    def get_gaze_bounds(self) -> tuple[float, float, float, float]:
        """Get pitch/yaw bounds of the calibration data.

        Returns:
            (pitch_min, pitch_max, yaw_min, yaw_max)
        """
        return calculate_gaze_bounds(self.calibration)
