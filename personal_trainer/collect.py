"""Face image collection during calibration for personalised gaze training.

Runs the calibration UI (reusing CalibrationInterface) but replaces the L2CS
gaze model with a face-detector-only capture. For each calibration point the
user focuses on, all detected face crops are saved to disk together with a
metadata CSV that records the known screen position of the target.

Output layout::

    <output_dir>/<session_id>/
        point_01/
            frame_0000.jpg
            frame_0001.jpg
            ...
        point_02/
            ...
        metadata.csv   # point_id, pixel_x, pixel_y, image_path
"""

import csv
import cv2
import numpy as np
import uuid
from batch_face.face_detection import RetinaFace
from calibration_tool.calibration import CalibrationInterface, PointState
from l2cs import select_device
from pathlib import Path
from typing import List, Optional, Tuple


class FaceCapture:
    """Webcam capture with face detection only — no L2CS gaze head."""

    def __init__(self, camera_id: int = 0, gpu: str = "0"):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        device = select_device(gpu)
        if device.type == "cpu":
            self.detector = RetinaFace()
        else:
            self.detector = RetinaFace(gpu_id=device.index)

    def capture_face_crop(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read one frame and return the highest-confidence face crop (RGB, 224×224).

        Returns:
            (False, None)     — camera read failed.
            (True, None)      — frame ok but no face detected.
            (True, crop_rgb)  — crop as uint8 RGB numpy array.
        """
        ret, frame = self.cap.read()
        if not ret:
            return False, None

        faces = self.detector(frame)
        if not faces:
            return True, None

        box, _landmark, _score = max(faces, key=lambda f: f[2])
        x_min = max(0, int(box[0]))
        y_min = max(0, int(box[1]))
        x_max = int(box[2])
        y_max = int(box[3])

        if x_max <= x_min or y_max <= y_min:
            return True, None

        crop = frame[y_min:y_max, x_min:x_max]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (224, 224))
        return True, crop

    def cleanup(self) -> None:
        self.cap.release()


class ImageCollectionSession:
    """Orchestrates calibration UI + FaceCapture to build a per-user training set."""

    def __init__(
        self,
        output_dir: Path,
        camera_id: int = 0,
        gpu: str = "0",
        fullscreen: bool = True,
        grid_size: int = 9,
        dwell_time: float = 8.0,
    ):
        self.session_id = str(uuid.uuid4())
        self.session_dir = output_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.interface = CalibrationInterface(fullscreen=fullscreen, grid_size=grid_size)
        # Override the class-level default so the user has time to slowly rotate
        # their head while keeping their eyes fixed on the calibration point.
        # This diversity of head poses is essential for a robust personalised model.
        self.interface.RECORDING_DURATION = dwell_time
        self.capture = FaceCapture(camera_id=camera_id, gpu=gpu)
        self._metadata: List[dict] = []

    def _point_dir(self, point_id: int) -> Path:
        d = self.session_dir / f"point_{point_id:02d}"
        d.mkdir(exist_ok=True)
        return d

    def run(self) -> Path:
        """Run the collection session.

        Returns:
            Path to the session directory (contains face crops + metadata.csv).
        """
        frame_counters: dict = {}

        try:
            while self.interface.run_frame():
                # run_frame() can return True on the tick that sets calibration_complete
                if self.interface.calibration_complete:
                    break

                current_pt = self.interface._get_current_point()
                if current_pt.state != PointState.RECORDING:
                    continue

                ok, crop = self.capture.capture_face_crop()
                if not ok or crop is None:
                    continue

                pid = current_pt.point_id
                n = frame_counters.get(pid, 0)
                frame_counters[pid] = n + 1

                img_path = self._point_dir(pid) / f"frame_{n:04d}.jpg"
                cv2.imwrite(str(img_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

                self._metadata.append(
                    {
                        "point_id": pid,
                        "pixel_x": current_pt.x,
                        "pixel_y": current_pt.y,
                        "image_path": str(img_path.relative_to(self.session_dir)),
                    }
                )
        finally:
            self.interface.cleanup()
            self.capture.cleanup()
            self._write_metadata()

        print(
            f"Collection complete — session {self.session_id}: "
            f"{len(self._metadata)} frames saved to {self.session_dir}"
        )
        return self.session_dir

    def _write_metadata(self) -> None:
        meta_path = self.session_dir / "metadata.csv"
        with open(meta_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["point_id", "pixel_x", "pixel_y", "image_path"])
            writer.writeheader()
            writer.writerows(self._metadata)
