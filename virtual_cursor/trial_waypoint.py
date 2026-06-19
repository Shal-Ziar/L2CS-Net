"""Simple square waypoint trial to validate cursor accuracy."""

import numpy as np
import time
from dataclasses import dataclass
from virtual_cursor.cursor_app import CursorApp


@dataclass
class TrialResult:
    """Result of a single waypoint trial."""

    waypoint_id: int
    target_pos: tuple[int, int]
    dwell_time: float
    pixel_errors: list[float]
    rms_error: float


class WaypointTrial:
    """4-corner square waypoint trial."""

    def __init__(
        self,
        cursor_app: CursorApp,
        dwell_threshold: float = 2.0,
        sample_radius: int = 50,
    ):
        """Initialize trial.

        Args:
            cursor_app: CursorApp instance
            dwell_threshold: Time in seconds to dwell on target before advancing
            sample_radius: Radius around target to count as "on target" in pixels
        """
        self.app = cursor_app
        self.dwell_threshold = dwell_threshold
        self.sample_radius = sample_radius

        # Screen dimensions
        sw, sh = cursor_app.gaze_to_pixel.screen_width, cursor_app.gaze_to_pixel.screen_height

        # Define 4-corner square waypoints
        margin = 100
        self.waypoints = [
            (margin, margin),
            (sw - margin, margin),
            (sw - margin, sh - margin),
            (margin, sh - margin),
        ]

        self.results: list[TrialResult] = []

    def is_on_target(self, cursor_pos: tuple[int, int], target_pos: tuple[int, int]) -> bool:
        """Check if cursor is within sample radius of target.

        Args:
            cursor_pos: Current cursor position
            target_pos: Target position

        Returns:
            True if within radius
        """
        dx = cursor_pos[0] - target_pos[0]
        dy = cursor_pos[1] - target_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        return distance <= self.sample_radius

    def draw_target(self, target_pos: tuple[int, int]) -> None:
        """Draw target circle on screen (in addition to cursor).

        Args:
            target_pos: Target position
        """
        import pygame

        # Draw filled circle at target
        pygame.draw.circle(
            self.app.screen,
            (100, 100, 255),
            target_pos,
            self.sample_radius,
        )
        # Draw ring around it
        pygame.draw.circle(
            self.app.screen,
            (150, 150, 255),
            target_pos,
            self.sample_radius,
            2,
        )

    def draw_current_waypoint_text(self, waypoint_idx: int) -> None:
        """Draw text showing current waypoint.

        Args:
            waypoint_idx: Index of current waypoint
        """
        import pygame

        font = pygame.font.Font(None, 36)
        text = font.render(f"Target {waypoint_idx + 1}/4", True, (255, 255, 255))
        self.app.screen.blit(text, (20, 20))

    def run_trial(self) -> list[TrialResult]:
        """Run 4-waypoint trial.

        Returns:
            List of TrialResult objects
        """
        import pygame

        print("\n=== STARTING WAYPOINT TRIAL ===")
        print("Look at each target in sequence. Dwell for 2 seconds to advance.")
        print("Press ESC to exit trial.\n")

        for waypoint_idx, target_pos in enumerate(self.waypoints):
            print(f"Waypoint {waypoint_idx + 1}/4: {target_pos}")

            dwell_start = None
            trial_errors = []

            while True:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return self.results
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return self.results

                # Process frame
                success, cursor_pos, valid = self.app.process_frame()
                if not success:
                    break

                # Clear screen
                self.app.screen.fill((20, 20, 20))

                # Draw target
                self.draw_target(target_pos)

                # Check if on target
                on_target = valid and self.is_on_target(cursor_pos, target_pos)

                if on_target:
                    if dwell_start is None:
                        dwell_start = time.time()
                    dwell_elapsed = time.time() - dwell_start

                    # Calculate error
                    dx = cursor_pos[0] - target_pos[0]
                    dy = cursor_pos[1] - target_pos[1]
                    error = np.sqrt(dx**2 + dy**2)
                    trial_errors.append(error)

                    # Display dwell progress
                    progress = min(dwell_elapsed / self.dwell_threshold, 1.0)
                    progress_color = (int(100 + 155 * progress), 100, 100)
                    pygame.draw.circle(
                        self.app.screen,
                        progress_color,
                        target_pos,
                        int(self.sample_radius * 0.7),
                    )

                    # Check if dwell complete
                    if dwell_elapsed >= self.dwell_threshold:
                        rms_error = float(np.sqrt(np.mean(np.array(trial_errors) ** 2)))
                        result = TrialResult(
                            waypoint_id=waypoint_idx + 1,
                            target_pos=target_pos,
                            dwell_time=dwell_elapsed,
                            pixel_errors=trial_errors,
                            rms_error=rms_error,
                        )
                        self.results.append(result)
                        print(f"  Completed! RMS error: {rms_error:.1f} px\n")
                        break

                else:
                    dwell_start = None

                # Render cursor on top
                self.app.render(cursor_pos, valid)
                self.draw_current_waypoint_text(waypoint_idx)

                # Display dwell requirement
                if dwell_start is not None:
                    font = pygame.font.Font(None, 28)
                    elapsed = time.time() - dwell_start
                    dwell_text = font.render(
                        f"Dwell: {elapsed:.1f}s / {self.dwell_threshold:.1f}s",
                        True,
                        (255, 255, 100),
                    )
                    self.app.screen.blit(dwell_text, (20, 60))

                pygame.display.flip()
                self.app.clock.tick(60)

        self.print_summary()
        return self.results

    def print_summary(self) -> None:
        """Print trial summary statistics."""
        if not self.results:
            print("No results to summarize.")
            return

        print("\n=== TRIAL SUMMARY ===")
        total_rms = []
        for result in self.results:
            total_rms.append(result.rms_error)
            print(
                f"  Waypoint {result.waypoint_id}: "
                f"RMS {result.rms_error:.1f} px, "
                f"Dwell {result.dwell_time:.2f}s"
            )

        mean_rms = float(np.mean(total_rms))
        std_rms = float(np.std(total_rms))
        print(f"\nMean RMS error: {mean_rms:.1f} ± {std_rms:.1f} px")
        print(f"Max RMS error: {max(total_rms):.1f} px")
        print(f"Min RMS error: {min(total_rms):.1f} px")
