"""Main virtual cursor application with gaze tracking."""

import cv2
import numpy as np
import pygame
import torch
from collections import deque
from l2cs import Pipeline
from pathlib import Path
from virtual_cursor.calibration_loader import CalibrationLoader
from virtual_cursor.gaze_to_pixel import GazeToPixel
from virtual_cursor.smoother import GazeSmoother


class CursorApp:
    """Real-time gaze-based cursor visualization."""

    def __init__(
        self,
        calibration_path: Path | str,
        model_path: Path | str | None = None,
        device: str = "cuda",
        camera_id: int = 0,
        fullscreen: bool = True,
        history_size: int = 10,
        arch: str = "ResNet50",
        fit_type: str = "univariate",
        smoother: GazeSmoother | None = None,
        ridge_alpha: float = 1.0,
    ):
        """Initialize cursor application.

        Args:
            calibration_path: Path to calibration JSONL file
            model_path: Path to gaze model weights (None uses default)
            device: Torch device ('cuda' or 'cpu')
            camera_id: Camera device ID
            fullscreen: Display in fullscreen mode
            history_size: Number of gaze samples to track for variance calculation
            arch: Model architecture (e.g., 'ResNet50')
            fit_type: "univariate" or "bivariate" calibration fit
            smoother: Optional pixel-space smoother applied after polynomial mapping.
                If None, the raw polynomial output is used without smoothing.
            ridge_alpha: Ridge regularization strength for bivariate fit (ignored for univariate).
        """
        # Load calibration
        print(f"Loading calibration from {calibration_path}...")
        self.calib_loader = CalibrationLoader(
            calibration_path, fit_type=fit_type, ridge_alpha=ridge_alpha
        )
        screen_w, screen_h = self.calib_loader.get_screen_dims()

        print(f"Screen: {screen_w}×{screen_h}")
        print(f"Fit type: {fit_type}")

        # Initialize gaze-to-pixel mapper with appropriate coefficients
        if fit_type == "univariate":
            poly_pitch_to_x, poly_yaw_to_y = self.calib_loader.get_polynomials()
            self.gaze_to_pixel = GazeToPixel(
                screen_w,
                screen_h,
                poly_pitch_to_x=poly_pitch_to_x,
                poly_yaw_to_y=poly_yaw_to_y,
                fit_type=fit_type,
            )
        else:  # bivariate
            coeffs_x, coeffs_y = self.calib_loader.get_bivariate_coeffs()
            self.gaze_to_pixel = GazeToPixel(
                screen_w, screen_h, coeffs_x=coeffs_x, coeffs_y=coeffs_y, fit_type=fit_type
            )

        # Load gaze model
        print("Loading gaze model...")
        if model_path is None:
            # Use default model
            model_path = (
                Path(__file__).parent.parent / "models" / "l2cs_gaze360_resnet50.safetensors"
            )
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Default model not found at {model_path}. "
                    "Specify --model to use a custom model path."
                )

        # Convert device string to torch device
        device_obj = self._parse_device(device)

        self.pipeline = Pipeline(
            weights=Path(model_path), arch=arch, device=device_obj, include_detector=True
        )

        # Open camera
        print(f"Opening camera {camera_id}...")
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")

        # Pygame setup
        pygame.init()
        flags = pygame.FULLSCREEN if fullscreen else 0
        self.screen = pygame.display.set_mode((screen_w, screen_h), flags=flags, vsync=1)
        pygame.display.set_caption("Virtual Cursor")
        self.clock = pygame.time.Clock()

        # Gaze history for variance tracking
        self.gaze_history: deque = deque(maxlen=history_size)
        self.last_cursor_pos: tuple[int, int] = (screen_w // 2, screen_h // 2)
        self.last_valid_pos: tuple[int, int] = (screen_w // 2, screen_h // 2)
        self.cursor_visible = True
        self.smoother = smoother

    def _parse_device(self, device_str: str) -> torch.device:
        """Parse device string to torch device.

        Args:
            device_str: 'cuda' or 'cpu'

        Returns:
            torch.device object
        """
        if device_str.lower() == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")

    def update_gaze_history(self, pitch: float, yaw: float) -> None:
        """Add gaze sample to history.

        Args:
            pitch: Pitch angle in radians
            yaw: Yaw angle in radians
        """
        self.gaze_history.append((pitch, yaw))

    def get_gaze_variance(self) -> float:
        """Calculate variance of recent gaze samples.

        Returns:
            Scalar variance (0 if insufficient samples)
        """
        if len(self.gaze_history) < 2:
            return 0.0

        samples = np.array(list(self.gaze_history))
        return float(np.var(samples))

    def process_frame(self) -> tuple[bool, tuple[int, int], bool]:
        """Capture and process single frame.

        Returns:
            (success, cursor_pos, valid_detection) tuple
            - success: True if frame captured
            - cursor_pos: (x, y) pixel coordinates
            - valid_detection: True if gaze detected
        """
        ret, frame = self.cap.read()
        if not ret:
            return False, self.last_valid_pos, False

        # Run gaze inference
        try:
            results = self.pipeline.step(frame)
        except Exception:
            return True, self.last_valid_pos, False

        # Check if any faces detected
        if results.pitch is None or len(results.pitch) == 0:
            return True, self.last_valid_pos, False

        # Use first detected face
        pitch = float(results.pitch[0])
        yaw = float(results.yaw[0])

        # Update gaze history
        self.update_gaze_history(pitch, yaw)

        # Map to pixel coordinates
        cursor_pos = self.gaze_to_pixel.gaze_to_pixel(pitch, yaw)

        # Apply optional smoother
        if self.smoother is not None:
            cursor_pos = self.smoother.update(*cursor_pos)

        self.last_cursor_pos = cursor_pos
        self.last_valid_pos = cursor_pos

        return True, cursor_pos, True

    def render(self, cursor_pos: tuple[int, int], valid: bool) -> None:
        """Render frame to pygame surface.

        Args:
            cursor_pos: (x, y) cursor position in pixels
            valid: True if gaze detection valid
        """
        # Clear screen
        self.screen.fill((20, 20, 20))

        # Calculate error ellipse size
        variance = self.get_gaze_variance()
        base_radius = 20
        error_scale = 1.0 + min(variance * 100, 3.0)  # Scale by variance, max 4x
        ellipse_radius = int(base_radius * error_scale)

        # Color based on detection state and variance
        if not valid:
            # Gray when no detection
            color = (128, 128, 128)
        elif variance > 0.01:
            # Red when high variance
            color = (255, 50, 50)
        else:
            # Green when stable
            color = (50, 255, 50)

        # Draw cursor ellipse
        pygame.draw.ellipse(
            self.screen,
            color,
            (
                cursor_pos[0] - ellipse_radius,
                cursor_pos[1] - ellipse_radius,
                ellipse_radius * 2,
                ellipse_radius * 2,
            ),
        )

        # Draw crosshair
        cross_size = 10
        pygame.draw.line(
            self.screen,
            (200, 200, 200),
            (cursor_pos[0] - cross_size, cursor_pos[1]),
            (cursor_pos[0] + cross_size, cursor_pos[1]),
            1,
        )
        pygame.draw.line(
            self.screen,
            (200, 200, 200),
            (cursor_pos[0], cursor_pos[1] - cross_size),
            (cursor_pos[0], cursor_pos[1] + cross_size),
            1,
        )

        # Update display
        pygame.display.flip()

    def run(self) -> None:
        """Main event loop."""
        print("Starting virtual cursor. Press ESC to exit.")

        try:
            while True:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return

                # Process frame
                success, cursor_pos, valid = self.process_frame()
                if not success:
                    break

                # Render
                self.render(cursor_pos, valid)
                self.clock.tick(60)  # Target 60 FPS display

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Release resources."""
        print("Cleaning up...")
        self.cap.release()
        pygame.quit()
