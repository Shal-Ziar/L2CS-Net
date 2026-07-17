import numpy as np
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from typing import List


class PointState(str, Enum):
    """State of a calibration point during the calibration process."""

    INACTIVE = "inactive"  # Red - not yet selected
    SELECTED = "selected"  # Green - selected, ready to record
    RECORDING = "recording"  # Blue - currently recording gaze data
    COMPLETED = "completed"  # Grey - recording done


class GazeResultContainer(BaseModel):
    """
    Container for the results of the gaze estimation. This is used to store the results of the gaze estimation and pass it to the rendering function.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pitch: np.ndarray = Field(..., description="The pitch of the gaze in radians")
    yaw: np.ndarray = Field(..., description="The yaw of the gaze in radians")
    bboxes: np.ndarray = Field(
        ...,
        description="The bounding boxes of the detected faces in the format [x_min, y_min, x_max, y_max]",
    )
    landmarks: np.ndarray = Field(
        ...,
        description="The landmarks of the detected faces in the format [x1, y1, x2, y2, ...] where x and y are the coordinates of the landmarks",
    )
    scores: np.ndarray = Field(..., description="The confidence scores of the detected faces")

    @field_validator("pitch", "yaw", "bboxes", "landmarks", "scores", mode="before")
    @classmethod
    def convert_to_array(cls, v):
        """Convert lists to numpy arrays during deserialization from JSON."""
        if isinstance(v, list):
            return np.array(v)
        return v

    @field_serializer("pitch", "yaw", "bboxes", "landmarks", "scores")
    def serialize_arrays(self, v):
        return v.tolist()


class CalibrationPoint(BaseModel):
    calibration_point: int = Field(
        ..., description="The index of the calibration point, from 1 to 9", ge=1, le=9
    )
    GazeResults: List[GazeResultContainer] = Field(
        ...,
        description="The gaze results for this calibration point, stored as a list of GazeResultContainer objects, one for each time the spacebar was pressed at this calibration point",
    )
    pixel_coordinates: List[int] = Field(
        ...,
        description="The pixel coordinates of the calibration point on the screen, stored as a list of [x, y] pairs, one for each time the spacebar was pressed at this calibration point",
    )
    session_type: str = Field(
        default="full",
        description="Type of calibration session: 'full' (9-point) or 'micro' (5-point hotkey)",
    )


class Calibration(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    timestamp: str = Field(..., description="Iso timestring of calibration time")
    screen_width: int = Field(..., description="Screen width in pixels")
    screen_height: int = Field(..., description="Screen height in pixels")
    calibration_points: List[CalibrationPoint] = []
    session_id: str = Field(
        default="",
        description="Unique ID grouping multiple calibration checkpoints from same session",
    )
