"""Main virtual cursor application with gaze tracking."""

import cv2
import numpy as np
import pygame
import time
import torch
import uuid
from calibration_tool.types import Calibration, CalibrationPoint, GazeResultContainer
from collections import deque
from datetime import datetime
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

        # Micro-calibration state
        self.calibration_path = calibration_path
        self.fit_type = fit_type
        self.ridge_alpha = ridge_alpha
        self.micro_calib_collecting = False
        self.micro_calib_points: list = []

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

    def trigger_micro_calibration(self) -> None:
        """Collect 5-point micro-calibration and update polynomials."""
        print("\n" + "=" * 60)
        print("MICRO-CALIBRATION: Collecting 5 points (1.5s each)")
        print("Look at each highlighted point on screen.")
        print("=" * 60 + "\n")

        self.micro_calib_collecting = True
        self.micro_calib_points = []

        # 5 points: center, corners
        screen_w, screen_h = self.calib_loader.get_screen_dims()
        points_5 = [
            (screen_w // 2, screen_h // 2),  # center
            (int(screen_w * 0.1), int(screen_h * 0.1)),  # top-left
            (int(screen_w * 0.9), int(screen_h * 0.1)),  # top-right
            (int(screen_w * 0.1), int(screen_h * 0.9)),  # bottom-left
            (int(screen_w * 0.9), int(screen_h * 0.9)),  # bottom-right
        ]

        # Collect 1.5s per point
        for point_idx, (target_x, target_y) in enumerate(points_5, 1):
            point_start = time.time()
            point_samples = []

            print(f"Point {point_idx}/5 at ({target_x}, {target_y})...")

            while (time.time() - point_start) < 1.5:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    continue

                try:
                    results = self.pipeline.step(frame)
                except Exception:
                    continue

                if results.pitch is None or len(results.pitch) == 0:
                    continue

                pitch = float(results.pitch[0])
                yaw = float(results.yaw[0])
                bboxes = results.bboxes
                landmarks = results.landmarks
                scores = results.scores

                # Create gaze result
                gaze_result = GazeResultContainer(
                    pitch=np.array([pitch]),
                    yaw=np.array([yaw]),
                    bboxes=bboxes,
                    landmarks=landmarks,
                    scores=scores,
                )
                point_samples.append(gaze_result)

                # Render collection UI
                self._render_calibration_point(target_x, target_y)
                self.clock.tick(30)

            if point_samples:
                self.micro_calib_points.append((point_idx, target_x, target_y, point_samples))
                print(f"  ✓ Collected {len(point_samples)} samples")
            else:
                print("  ✗ No gaze detected")

        self.micro_calib_collecting = False

        # Append to calibration file
        self._append_micro_calibration()
        print("Micro-calibration complete. Reloading...\n")

        # Reload calibration and update polynomials
        self._reload_calibration()

    def _render_calibration_point(self, x: int, y: int) -> None:
        """Render calibration collection UI."""
        self.screen.fill((20, 20, 20))

        # Draw large circle at target point
        pygame.draw.circle(self.screen, (0, 255, 0), (x, y), 40, 3)
        pygame.draw.circle(self.screen, (0, 255, 0), (x, y), 20)

        # Crosshair
        pygame.draw.line(self.screen, (200, 200, 200), (x - 50, y), (x + 50, y), 1)
        pygame.draw.line(self.screen, (200, 200, 200), (x, y - 50), (x, y + 50), 1)

        # Info text
        font = pygame.font.Font(None, 24)
        text = font.render("Collecting calibration data...", True, (200, 200, 200))
        self.screen.blit(text, (10, 10))

        pygame.display.flip()

    def _append_micro_calibration(self) -> None:
        """Append micro-calibration data to calibration file."""
        try:
            # Load existing calibration
            with open(self.calibration_path, "r") as f:
                existing = Calibration.model_validate_json(f.read())

            session_id = existing.session_id or str(uuid.uuid4())

            # Create new points
            for point_id, x, y, gaze_results in self.micro_calib_points:
                cal_point = CalibrationPoint(
                    calibration_point=point_id,
                    GazeResults=gaze_results,
                    pixel_coordinates=[x, y],
                    session_type="micro",
                )
                existing.calibration_points.append(cal_point)

            existing.session_id = session_id
            existing.timestamp = datetime.now().isoformat()

            # Write back
            with open(self.calibration_path, "w") as f:
                f.write(existing.model_dump_json(indent=3))

            print(f"✓ Appended to {self.calibration_path}")
        except Exception as e:
            print(f"Error appending calibration: {e}")

    def _reload_calibration(self) -> None:
        """Reload calibration and update polynomials."""
        try:
            self.calib_loader = CalibrationLoader(
                self.calibration_path,
                fit_type=self.fit_type,
                ridge_alpha=self.ridge_alpha,
            )

            screen_w, screen_h = self.calib_loader.get_screen_dims()

            if self.fit_type == "univariate":
                poly_pitch_to_x, poly_yaw_to_y = self.calib_loader.get_polynomials()
                self.gaze_to_pixel = GazeToPixel(
                    screen_w,
                    screen_h,
                    poly_pitch_to_x=poly_pitch_to_x,
                    poly_yaw_to_y=poly_yaw_to_y,
                    fit_type=self.fit_type,
                )
            else:  # bivariate
                coeffs_x, coeffs_y = self.calib_loader.get_bivariate_coeffs()
                self.gaze_to_pixel = GazeToPixel(
                    screen_w, screen_h, coeffs_x=coeffs_x, coeffs_y=coeffs_y, fit_type=self.fit_type
                )

            print("✓ Calibration reloaded and polynomials updated")
        except Exception as e:
            print(f"Error reloading calibration: {e}")

    def run(self) -> None:
        """Main event loop."""
        print("Starting virtual cursor. Press ESC to exit, 'c' to recalibrate.")

        try:
            while True:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return
                        if event.key == pygame.K_c and not self.micro_calib_collecting:
                            self.trigger_micro_calibration()

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
