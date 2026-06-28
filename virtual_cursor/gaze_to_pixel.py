"""Map gaze angles (pitch, yaw in radians) to screen pixels."""

import numpy as np


class GazeToPixel:
    """Convert gaze angles to screen coordinates using polynomial calibration.

    Note: The calibration mapping is:
    - Pitch (head tilt) → Pixel X (horizontal)
    - Yaw (head turn) → Pixel Y (vertical)

    This is counter-intuitive but matches the actual calibration data.
    """

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        poly_pitch_to_x: np.ndarray | None = None,
        poly_yaw_to_y: np.ndarray | None = None,
        coeffs_x: np.ndarray | None = None,
        coeffs_y: np.ndarray | None = None,
        fit_type: str = "univariate",
        pitch_bounds: tuple[float, float] | None = None,
        yaw_bounds: tuple[float, float] | None = None,
    ):
        """Initialize gaze-to-pixel mapper.

        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            poly_pitch_to_x: Univariate polynomial for pitch → pixel_x (required if fit_type='univariate')
            poly_yaw_to_y: Univariate polynomial for yaw → pixel_y (required if fit_type='univariate')
            coeffs_x: Bivariate coefficients for pixel_x (required if fit_type='bivariate')
            coeffs_y: Bivariate coefficients for pixel_y (required if fit_type='bivariate')
            fit_type: "univariate" or "bivariate"
            pitch_bounds: (min, max) pitch range from calibration. Input is clamped to this range
                before polynomial evaluation to prevent extrapolation divergence.
            yaw_bounds: (min, max) yaw range from calibration. Same purpose as pitch_bounds.
        """
        if fit_type not in ("univariate", "bivariate"):
            raise ValueError(f"Invalid fit_type: {fit_type}. Use 'univariate' or 'bivariate'.")

        self.fit_type = fit_type
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.pitch_bounds = pitch_bounds
        self.yaw_bounds = yaw_bounds

        if fit_type == "univariate":
            if poly_pitch_to_x is None or poly_yaw_to_y is None:
                raise ValueError("univariate fit_type requires poly_pitch_to_x and poly_yaw_to_y")
            self.poly_pitch_to_x = poly_pitch_to_x
            self.poly_yaw_to_y = poly_yaw_to_y
            self.coeffs_x = None
            self.coeffs_y = None
        else:  # bivariate
            if coeffs_x is None or coeffs_y is None:
                raise ValueError("bivariate fit_type requires coeffs_x and coeffs_y")
            self.coeffs_x = coeffs_x
            self.coeffs_y = coeffs_y
            self.poly_pitch_to_x = None
            self.poly_yaw_to_y = None

    def _eval_bivariate(self, pitch: float, yaw: float, coeffs: np.ndarray) -> float:
        """Evaluate 2D polynomial with cross-terms.

        Args:
            pitch: Pitch angle
            yaw: Yaw angle
            coeffs: Bivariate coefficients (length 10)

        Returns:
            Evaluated polynomial value
        """
        A = np.array(
            [
                1.0,
                pitch,
                yaw,
                pitch**2,
                pitch * yaw,
                yaw**2,
                pitch**3,
                pitch**2 * yaw,
                pitch * yaw**2,
                yaw**3,
            ]
        )
        return float(A @ coeffs)

    def gaze_to_pixel(self, pitch: float, yaw: float) -> tuple[int, int]:
        """Map gaze angles to screen coordinates.

        Args:
            pitch: Pitch angle in radians (head tilt)
            yaw: Yaw angle in radians (head turn)

        Returns:
            (pixel_x, pixel_y) tuple, clamped to screen bounds
        """
        # Clamp input to calibrated gaze range to prevent polynomial extrapolation divergence
        if self.pitch_bounds is not None:
            pitch = float(np.clip(pitch, *self.pitch_bounds))
        if self.yaw_bounds is not None:
            yaw = float(np.clip(yaw, *self.yaw_bounds))

        # Evaluate based on fit type
        if self.fit_type == "univariate":
            pixel_x = float(np.polyval(self.poly_pitch_to_x, pitch))
            pixel_y = float(np.polyval(self.poly_yaw_to_y, yaw))
        else:  # bivariate
            pixel_x = self._eval_bivariate(pitch, yaw, self.coeffs_x)
            pixel_y = self._eval_bivariate(pitch, yaw, self.coeffs_y)

        # Clamp to screen bounds
        pixel_x = max(0, min(int(pixel_x), self.screen_width - 1))
        pixel_y = max(0, min(int(pixel_y), self.screen_height - 1))

        return (pixel_x, pixel_y)

    def batch_gaze_to_pixel(
        self, pitch_arr: np.ndarray, yaw_arr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map batch of gaze angles to screen coordinates.

        Args:
            pitch_arr: Array of pitch angles in radians
            yaw_arr: Array of yaw angles in radians

        Returns:
            (pixel_x_arr, pixel_y_arr) tuple, clamped to screen bounds
        """
        if self.fit_type == "univariate":
            pixel_x = np.polyval(self.poly_pitch_to_x, pitch_arr).astype(int)
            pixel_y = np.polyval(self.poly_yaw_to_y, yaw_arr).astype(int)
        else:  # bivariate
            A_x = np.column_stack(
                [
                    np.ones_like(pitch_arr),
                    pitch_arr,
                    yaw_arr,
                    pitch_arr**2,
                    pitch_arr * yaw_arr,
                    yaw_arr**2,
                    pitch_arr**3,
                    pitch_arr**2 * yaw_arr,
                    pitch_arr * yaw_arr**2,
                    yaw_arr**3,
                ]
            )
            pixel_x = (A_x @ self.coeffs_x).astype(int)
            pixel_y = (A_x @ self.coeffs_y).astype(int)

        # Clamp to screen bounds
        pixel_x = np.clip(pixel_x, 0, self.screen_width - 1)
        pixel_y = np.clip(pixel_y, 0, self.screen_height - 1)

        return (pixel_x, pixel_y)
