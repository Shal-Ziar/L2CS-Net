"""9-point calibration tool for L2CS gaze detection.

Calibration steps:
1. Start gaze model
2. Create pygame fullscreen interface
3. Display 9 points in 3x3 grid (red circles with order numbers)
4. Cycle through points on spacebar:
   - First press: select point (turn green)
   - Second press: start recording for 2 seconds (turn blue)
   - After 2 seconds: finalize point (turn grey)
5. Store calibration data as JSONL with CalibrationPoint entries

Point layout:
1. top-left    2. top-center    3. top-right
4. mid-left    5. center        6. mid-right
7. bot-left    8. bot-center    9. bot-right

Interface: Full-screen, auto-scales based on screen size.
"""

import cv2
import json
import numpy as np
import pygame
import time
from calibration_tool.types import CalibrationPoint, GazeResultContainer, PointState
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class CalibrationPoint3D:
    """Represents a single calibration point on the screen with state and data."""

    def __init__(self, point_id: int, x: int, y: int):
        self.point_id = point_id  # 1-9
        self.x = x
        self.y = y
        self.state = PointState.INACTIVE
        self.gaze_results: List[GazeResultContainer] = []
        self.selection_time: float = 0.0  # Time when point was selected

    def reset(self) -> None:
        """Reset point to inactive state."""
        self.state = PointState.INACTIVE
        self.gaze_results = []
        self.selection_time = 0.0


class CalibrationInterface:
    """Main calibration interface with pygame rendering and state management."""

    # Color definitions (RGB)
    COLOR_WHITE = (255, 255, 255)
    COLOR_RED = (255, 0, 0)
    COLOR_GREEN = (0, 255, 0)
    COLOR_BLUE = (0, 0, 255)
    COLOR_GREY = (128, 128, 128)
    COLOR_BLACK = (0, 0, 0)

    POINT_RADIUS = 30  # Radius of calibration point circle
    POINT_NUMBER_FONT_SIZE = 24
    CROSSHAIR_SIZE = 15  # Size of crosshair arms
    RECORDING_DURATION = 2.0  # Seconds to record per point

    def __init__(self, fullscreen: bool = True):
        """
        Initialize the calibration interface.

        Args:
            fullscreen: If True, use fullscreen mode; else use windowed mode
        """
        pygame.init()

        # Create display in fullscreen mode to get screen dimensions
        if fullscreen:
            flags = pygame.FULLSCREEN  # | pygame.SCALED
            self.screen = pygame.display.set_mode((0, 0), flags)
        else:
            # Windowed mode with default size
            self.screen_width = 1280
            self.screen_height = 720
            flags = 0
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), flags)

        # Get actual screen dimensions
        self.screen_width = self.screen.get_width()
        self.screen_height = self.screen.get_height()

        pygame.display.set_caption("L2CS Gaze Calibration")

        # Initialize font
        self.font_numbers = pygame.font.Font(None, self.POINT_NUMBER_FONT_SIZE)
        self.font_info = pygame.font.Font(None, 24)

        # Create calibration points in 3x3 grid
        self.calibration_points = self._create_grid_points()
        self.current_point_idx = 0  # Index into ordered point list
        self.calibration_data: Dict[int, CalibrationPoint3D] = {
            pt.point_id: pt for pt in self.calibration_points
        }

        self.running = True
        self.calibration_complete = False
        self.clock = pygame.time.Clock()
        self.fps = 30

    def _create_grid_points(self) -> List[CalibrationPoint3D]:
        """Create 9 calibration points in a 3x3 grid with margins."""
        margin_x = int(self.screen_width * 0.1)
        margin_y = int(self.screen_height * 0.1)

        usable_width = self.screen_width - 2 * margin_x
        usable_height = self.screen_height - 2 * margin_y

        # Spacing for 3x3 grid
        step_x = usable_width // 2
        step_y = usable_height // 2

        points = []
        point_id = 1
        for row in range(3):
            for col in range(3):
                x = margin_x + col * step_x
                y = margin_y + row * step_y
                points.append(CalibrationPoint3D(point_id, x, y))
                point_id += 1

        return points

    def _get_current_point(self) -> CalibrationPoint3D:
        """Get the currently active calibration point."""
        return self.calibration_points[self.current_point_idx]

    def _advance_to_next_point(self):
        """
        Advance to the next calibration point.

        Returns:
            True if there are more points, False if calibration is complete
        """
        self.current_point_idx += 1
        if self.current_point_idx >= len(self.calibration_points):
            self.calibration_complete = True
            self.running = False

    def handle_spacebar_press(self) -> None:
        """Handle spacebar press to advance point state."""
        current_pt = self._get_current_point()

        if current_pt.state == PointState.INACTIVE:
            # First press: select point
            current_pt.state = PointState.SELECTED
        elif current_pt.state == PointState.SELECTED:
            # Second press: start recording
            current_pt.state = PointState.RECORDING
            current_pt.selection_time = time.time()

    def handle_esc_press(self) -> None:
        """Handle ESC press to exit calibration."""
        self.running = False

    def process_events(self) -> None:
        """Process pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not self.calibration_complete:
                    self.handle_spacebar_press()
                if event.key == pygame.K_ESCAPE:
                    self.handle_esc_press()

    def update(self) -> None:
        """Update calibration state (check recording timers, etc.)."""
        current_pt = self._get_current_point()

        if current_pt.state == PointState.RECORDING:
            elapsed = time.time() - current_pt.selection_time
            if elapsed >= self.RECORDING_DURATION:
                # Recording done
                current_pt.state = PointState.COMPLETED
                self._advance_to_next_point()

    def add_gaze_result(self, result: GazeResultContainer) -> None:
        """
        Add a gaze result to the current recording point.

        Args:
            result: GazeResultContainer from the gaze model
        """
        current_pt = self._get_current_point()
        if current_pt.state == PointState.RECORDING:
            current_pt.gaze_results.append(result)

    def _draw_point(self, pt: CalibrationPoint3D) -> None:
        """Draw a single calibration point with appropriate color and number."""
        # Determine color based on state
        if pt.state == PointState.INACTIVE:
            color = self.COLOR_RED
        elif pt.state == PointState.SELECTED:
            color = self.COLOR_GREEN
        elif pt.state == PointState.RECORDING:
            color = self.COLOR_BLUE
        else:  # COMPLETED
            color = self.COLOR_GREY

        # Draw circle
        pygame.draw.circle(self.screen, color, (pt.x, pt.y), self.POINT_RADIUS)

        # Draw crosshair
        pygame.draw.line(
            self.screen,
            self.COLOR_BLACK,
            (pt.x - self.CROSSHAIR_SIZE, pt.y),
            (pt.x + self.CROSSHAIR_SIZE, pt.y),
            2,
        )
        pygame.draw.line(
            self.screen,
            self.COLOR_BLACK,
            (pt.x, pt.y - self.CROSSHAIR_SIZE),
            (pt.x, pt.y + self.CROSSHAIR_SIZE),
            2,
        )

        # Draw point number
        number_text = self.font_numbers.render(str(pt.point_id), True, self.COLOR_BLACK)
        text_rect = number_text.get_rect(center=(pt.x, pt.y))
        self.screen.blit(number_text, text_rect)

    def _draw_center_crosshair(self) -> None:
        """Draw a crosshair at the center of the screen."""
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2

        pygame.draw.line(
            self.screen,
            self.COLOR_BLACK,
            (center_x - self.CROSSHAIR_SIZE, center_y),
            (center_x + self.CROSSHAIR_SIZE, center_y),
            1,
        )
        pygame.draw.line(
            self.screen,
            self.COLOR_BLACK,
            (center_x, center_y - self.CROSSHAIR_SIZE),
            (center_x, center_y + self.CROSSHAIR_SIZE),
            1,
        )

    def _draw_info_text(self) -> None:
        """Draw info text on screen."""
        if not self.calibration_complete:
            current_pt = self._get_current_point()
            info_text = self.font_info.render(
                f"Point {current_pt.point_id}/9 | Status: {current_pt.state.value} | Press SPACE to advance, ESC to exit",
                True,
                self.COLOR_BLACK,
            )
            self.screen.blit(info_text, (10, 10))
        else:
            complete_text = self.font_info.render(
                "Calibration Complete! Press ESC to exit.", True, self.COLOR_GREEN
            )
            self.screen.blit(complete_text, (10, 10))

    def render(self) -> None:
        """Render the calibration interface."""
        # Clear screen
        self.screen.fill(self.COLOR_WHITE)

        # Draw all points
        for pt in self.calibration_points:
            self._draw_point(pt)

        # Draw center crosshair
        self._draw_center_crosshair()

        # Draw info text
        self._draw_info_text()

        # Update display
        pygame.display.flip()

    def run_frame(self) -> bool:
        """
        Execute one frame of the calibration loop.

        Returns:
            False if the loop should exit, True otherwise
        """
        self.process_events()
        if not self.running:
            return False

        self.update()
        self.render()
        self.clock.tick(self.fps)

        return True

    def save_calibration(self, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Save collected calibration data to JSONL file.

        Args:
            output_path: Path to save JSONL file. If None, uses default location with timestamp.

        Returns:
            Path to saved file, or None if no data was collected
        """
        # Check if we have any data
        has_data = any(pt.gaze_results for pt in self.calibration_points)
        if not has_data:
            print("No calibration data collected.")
            return None

        # Create default output path if not provided
        if output_path is None:
            data_dir = Path(__file__).parent / "callibration_data"
            data_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = data_dir / f"calibration_{timestamp}.jsonl"

        # Create metadata header
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "type": "calibration_metadata",
        }

        # Write JSONL file
        with open(output_path, "w") as f:
            # Write metadata header
            f.write(json.dumps(metadata) + "\n")

            # Write calibration points
            for pt in self.calibration_points:
                if pt.gaze_results:  # Only write points with data
                    cal_point = CalibrationPoint(
                        calibration_point=pt.point_id, GazeResults=pt.gaze_results
                    )
                    # Custom serialization to handle numpy arrays
                    cal_dict = cal_point.model_dump()
                    # Convert numpy arrays to lists for JSON serialization
                    for gaze_result in cal_dict["GazeResults"]:
                        for key in ["pitch", "yaw", "bboxes", "landmarks", "scores"]:
                            if isinstance(gaze_result[key], np.ndarray):
                                gaze_result[key] = gaze_result[key].tolist()

                    f.write(json.dumps(cal_dict) + "\n")

        print(f"Calibration data saved to {output_path}")
        return output_path

    def cleanup(self) -> None:
        """Clean up pygame and resources."""
        pygame.quit()


class GazeCapture:
    """Handles gaze model initialization and frame capture."""

    def __init__(
        self, model_path: Optional[Path] = None, device: str = "gpu:0", camera_id: int = 0
    ):
        """
        Initialize gaze capture pipeline.

        Args:
            model_path: Path to L2CS model weights. If None, uses default.
            device: Device to use for inference (e.g., "gpu:0", "cpu")
            camera_id: Webcam device ID (0 for default)
        """
        # Import here to avoid hard dependency
        try:
            from l2cs import Pipeline, select_device
        except ImportError:
            raise ImportError("L2CS package required. Install with: pip install l2cs")

        # Set default model path if not provided
        if model_path is None:
            model_path = Path(__file__).parent.parent / "models" / "L2CSNet_gaze360.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Initialize gaze pipeline
        print(f"Loading gaze model from {model_path}...")
        self.pipeline = Pipeline(
            weights=model_path,
            arch="ResNet50",
            device=select_device(device, batch_size=1),
            include_detector=True,
            confidence_threshold=0.5,
        )
        print("Gaze model loaded successfully.")

        # Initialize webcam
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera (ID: {camera_id})")

        # Set camera properties for optimal performance
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to get fresh frames
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Camera initialized: {self.frame_width}x{self.frame_height} @ 30 FPS")

    def capture_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[GazeResultContainer]]:
        """
        Capture and process a single frame from the camera.

        Returns:
            Tuple of (success, frame, gaze_result)
            - success: True if frame was captured and processed
            - frame: The captured frame (BGR)
            - gaze_result: GazeResultContainer or None if no face detected
        """
        ret, frame = self.cap.read()
        if not ret:
            return False, None, None

        try:
            # Process frame with gaze model
            gaze_result = self.pipeline.step(frame)

            # Check if face was detected
            if gaze_result.bboxes.size == 0:
                # No face detected
                return True, frame, None

            return True, frame, gaze_result
        except Exception as e:
            print(f"Error processing frame: {e}")
            return False, None, None

    def cleanup(self) -> None:
        """Release camera resources."""
        self.cap.release()


class CalibrationSession:
    """Manages a complete calibration session combining interface and gaze capture."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "gpu:0",
        camera_id: int = 0,
        fullscreen: bool = True,
    ):
        """
        Initialize a calibration session.

        Args:
            model_path: Path to L2CS model weights
            device: Device for inference
            camera_id: Webcam device ID
            fullscreen: Use fullscreen display
        """
        self.interface = CalibrationInterface(fullscreen=fullscreen)
        self.gaze_capture = GazeCapture(model_path=model_path, device=device, camera_id=camera_id)

    def run(self) -> Optional[Path]:
        """
        Run the complete calibration session.

        Returns:
            Path to saved calibration data, or None if aborted
        """
        print("Calibration session started. Press SPACE to advance, ESC to exit.")

        try:
            while self.interface.running:
                # Capture gaze frame
                success, frame, gaze_result = self.gaze_capture.capture_frame()

                if not success:
                    print("Warning: Failed to capture frame")
                    continue

                # Add gaze result if face detected and recording
                if gaze_result is not None:
                    self.interface.add_gaze_result(gaze_result)

                # Run one frame of calibration interface
                if not self.interface.run_frame():
                    break

            # Save calibration data if we have any
            if not self.interface.calibration_complete:
                # User pressed ESC - abort without saving
                print("Calibration aborted or incomplete. No data saved.")
                return None

            output_path = self.interface.save_calibration()

            # Generate visualizations
            if output_path:
                try:
                    from calibration_tool.visualize import visualize_calibration

                    visualize_calibration(output_path)
                except Exception as e:
                    print(f"Warning: Failed to generate visualizations: {e}")

            print("Calibration completed and saved.")
            return output_path

        except KeyboardInterrupt:
            print("Calibration interrupted by user.")
            return None
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up all resources."""
        self.gaze_capture.cleanup()
        self.interface.cleanup()


if __name__ == "__main__":
    # Example usage
    session = CalibrationSession(fullscreen=True)
    session.run()
