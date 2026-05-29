import math
from enum import Enum
from typing import List
import numpy as np

from pydantic import BaseModel, Field, ConfigDict


class PointState(str, Enum):
    """State of a calibration point during the calibration process."""
    INACTIVE = "inactive"      # Red - not yet selected
    SELECTED = "selected"      # Green - selected, ready to record
    RECORDING = "recording"    # Blue - currently recording gaze data
    COMPLETED = "completed"    # Grey - recording done

class GazeSituationSettings(BaseModel):
    """
    Configuration that defines the setup. Ie distance from head to camera, screen size and curvature.
    """
    ScreenWidth: int = Field(..., description="The width of the screen in pixels")
    ScreenHeight: int = Field(..., description="The Height of the screen in pixels")
    DistanceToScreen: int= Field(..., description="Euclidian distance to camera from nose-bridge")
    ScreenCurvature: float=Field(default=math.inf, description="Screen curvature in mm, default inf for flatscreens")

class GazeResultContainer(BaseModel):
    """
    Container for the results of the gaze estimation. This is used to store the results of the gaze estimation and pass it to the rendering function.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    pitch: np.ndarray = Field(..., description="The pitch of the gaze in radians")
    yaw: np.ndarray = Field(..., description="The yaw of the gaze in radians")
    bboxes: np.ndarray = Field(..., description="The bounding boxes of the detected faces in the format [x_min, y_min, x_max, y_max]")
    landmarks: np.ndarray = Field(..., description="The landmarks of the detected faces in the format [x1, y1, x2, y2, ...] where x and y are the coordinates of the landmarks")
    scores: np.ndarray = Field(..., description="The confidence scores of the detected faces")

class CalibrationPoint(BaseModel):
    calibration_point: int = Field(..., description="The index of the calibration point, from 1 to 9", ge=1, le=9)
    GazeResults: List[GazeResultContainer] = Field(..., description="The gaze results for this calibration point, stored as a list of GazeResultContainer objects, one for each time the spacebar was pressed at this calibration point")

