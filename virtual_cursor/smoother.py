"""Cursor position smoothers for gaze-based cursor applications."""

from collections import deque
from typing import Protocol, runtime_checkable


@runtime_checkable
class GazeSmoother(Protocol):
    """Protocol for pixel-space cursor smoothers.

    Implementations receive raw pixel coordinates each frame and return a
    (possibly smoothed) position. Any stateful filter—moving average,
    exponential moving average, Kalman filter, etc.—can satisfy this protocol.
    """

    def update(self, x: int, y: int) -> tuple[int, int]:
        """Process a new cursor position and return the smoothed position.

        Args:
            x: Raw cursor x coordinate in pixels
            y: Raw cursor y coordinate in pixels

        Returns:
            Smoothed (x, y) pixel coordinates
        """
        ...

    def reset(self) -> None:
        """Clear internal state (e.g. call when tracking is lost)."""
        ...


class MovingAverageSmoother:
    """Simple frame-based moving average over pixel coordinates.

    Averages the last ``window`` cursor positions element-wise.  A window of 1
    is a no-op (returns the raw position unchanged).
    """

    def __init__(self, window: int) -> None:
        """Initialise the smoother.

        Args:
            window: Number of frames to average over (must be >= 1).
        """
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self._window = window
        self._history: deque[tuple[int, int]] = deque(maxlen=window)

    def update(self, x: int, y: int) -> tuple[int, int]:
        """Append position and return the mean over the current window.

        Args:
            x: Raw cursor x coordinate in pixels
            y: Raw cursor y coordinate in pixels

        Returns:
            Element-wise integer mean of the buffered positions
        """
        self._history.append((x, y))
        avg_x = int(sum(pos[0] for pos in self._history) / len(self._history))
        avg_y = int(sum(pos[1] for pos in self._history) / len(self._history))
        return avg_x, avg_y

    def reset(self) -> None:
        """Clear the position history."""
        self._history.clear()
