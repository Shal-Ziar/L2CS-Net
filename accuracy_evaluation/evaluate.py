"""Reusable core for static-point gaze-tracking trials: point grids, JSONL trial
recording (`TrialRecorder`), mock/live trial runners, pixel mapping, and metrics.

See `accuracy_evaluation.cli` for the `mock`/`live`/`aggregate` command-line interface.
"""

from __future__ import annotations

import json
import math
import numpy as np
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

CONDITIONS = (
    "no_calibration",
    "polynomial_univariate",
    "polynomial_bivariate",
    "post_training",
    "ring_light",
)

# (x, y, w, h) placeholder used by the mock predictor when no real face
# detector is wired in via `face_bbox_fn`.
_MOCK_FACE_BBOX_FRACTION = (0.4, 0.25, 0.2, 0.3)


def generate_static_points(
    screen_w: int, screen_h: int, rows: int = 3, cols: int = 3, margin: float = 0.12
) -> np.ndarray:
    """Deterministic grid of (x, y) pixel points, same order for every call/condition."""
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be >= 1")
    xs = np.linspace(screen_w * margin, screen_w * (1 - margin), cols)
    ys = np.linspace(screen_h * margin, screen_h * (1 - margin), rows)
    return np.array([[int(x), int(y)] for y in ys for x in xs], dtype=int)


def write_jsonl_lines(path: Path, records: list[dict]) -> None:
    """Write all records to `path` in one pass (overwrites any existing file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def head_size_from_bbox(
    face_bbox: tuple[int, int, int, int], screen_diag_px: float
) -> tuple[float, float]:
    """Return (head_size_px, head_size_norm), a screen-resolution-independent proximity proxy."""
    _, _, w, h = face_bbox
    head_size_px = math.sqrt(max(1, w * h))
    return head_size_px, head_size_px / max(1.0, screen_diag_px)


@dataclass
class TrialRecorder:
    """Accumulates per-frame JSONL rows for one trial and writes them in a single pass.

    Shared by `run_mock_trial` and `run_live_trial` so the JSONL schema only lives in one place.
    """

    session_id: str
    trial_id: str
    rep: int
    condition: str
    screen_size: tuple[int, int]
    model_version: str
    records: list[dict] = field(default_factory=list)

    @property
    def screen_diag(self) -> float:
        return math.hypot(*self.screen_size)

    def add(
        self,
        point_index: int,
        target: tuple[int, int],
        pred_xy: tuple[float, float],
        pred_angles: tuple[float, float],
        face_bbox: tuple[int, int, int, int],
        brightness_mean: float,
    ) -> None:
        x, y, w, h = face_bbox
        head_size_px, head_size_norm = head_size_from_bbox(face_bbox, self.screen_diag)
        self.records.append(
            {
                "session_id": self.session_id,
                "trial_id": self.trial_id,
                "rep": self.rep,
                "condition": self.condition,
                "timestamp": time.time(),
                "frame_idx": len(self.records),
                "point_index": point_index,
                "target_x_px": int(target[0]),
                "target_y_px": int(target[1]),
                "pred_x_px": float(pred_xy[0]),
                "pred_y_px": float(pred_xy[1]),
                "pred_pitch": float(pred_angles[0]),
                "pred_yaw": float(pred_angles[1]),
                "face_bbox": {"x": x, "y": y, "w": w, "h": h},
                "face_area": w * h,
                "face_height_px": h,
                "face_size_norm": h / float(self.screen_size[1]),
                "head_size_px": head_size_px,
                "head_size_norm": head_size_norm,
                "brightness_mean": brightness_mean,
                "camera_meta": {},
                "model_version": self.model_version,
                "extras": {},
            }
        )

    def write(self, path: Path) -> Path:
        write_jsonl_lines(path, self.records)
        return path


def _default_face_bbox(screen_w: int, screen_h: int) -> tuple[int, int, int, int]:
    """Fixed placeholder bbox for mock trials; replace with a real face detector output."""
    fx, fy, fw, fh = _MOCK_FACE_BBOX_FRACTION
    return int(screen_w * fx), int(screen_h * fy), int(screen_w * fw), int(screen_h * fh)


def simulate_model_predict(
    target_x: int, target_y: int, head_size_norm: float, rng: np.random.Generator
) -> tuple[float, float, float, float]:
    """Mock predictor: pixel noise decreases as head_size_norm grows (closer to camera)."""
    sigma = max(3.0, 60.0 * (0.5 - head_size_norm))
    pred_x = float(target_x + rng.normal(scale=sigma))
    pred_y = float(target_y + rng.normal(scale=sigma))
    return pred_x, pred_y, float(rng.normal(scale=0.01)), float(rng.normal(scale=0.01))


def run_mock_trial(
    session_id: str,
    trial_id: str,
    rep: int,
    condition: str,
    points: np.ndarray,
    out_path: Path,
    fps: int = 30,
    dwell: float = 1.5,
    model_predict: Callable[[int, int, float], tuple[float, float, float, float]] | None = None,
    face_bbox_fn: Callable[[], tuple[int, int, int, int]] | None = None,
    screen_size: tuple[int, int] = (1920, 1080),
    model_version: str = "mock",
    seed: int | None = None,
) -> Path:
    """Simulate a static-point trial (no camera/model) and write per-frame JSONL to `out_path`.

    `model_predict(target_x, target_y, head_size_norm) -> (pred_x, pred_y, pred_pitch, pred_yaw)`
    and `face_bbox_fn() -> (x, y, w, h)` default to a mock predictor/placeholder bbox; `seed`
    makes the mock predictor and brightness reproducible.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}")
    if dwell <= 0 or fps <= 0:
        raise ValueError("dwell and fps must be > 0")

    screen_w, screen_h = screen_size
    rng = np.random.default_rng(seed)
    face_bbox_fn = face_bbox_fn or (lambda: _default_face_bbox(screen_w, screen_h))
    model_predict = model_predict or (lambda tx, ty, hsn: simulate_model_predict(tx, ty, hsn, rng))
    recorder = TrialRecorder(session_id, trial_id, rep, condition, screen_size, model_version)

    frames_per_point = int(max(1, round(dwell * fps)))
    for point_index, (tx, ty) in enumerate(points):
        for _ in range(frames_per_point):
            face_bbox = face_bbox_fn()
            _, head_size_norm = head_size_from_bbox(face_bbox, recorder.screen_diag)
            pred_x, pred_y, pred_pitch, pred_yaw = model_predict(tx, ty, head_size_norm)
            recorder.add(
                point_index,
                (tx, ty),
                (pred_x, pred_y),
                (pred_pitch, pred_yaw),
                face_bbox,
                brightness_mean=float(100.0 + rng.normal(scale=5.0)),
            )

    return recorder.write(out_path)


def _read_jsonl(log_path: Path) -> list[dict]:
    with open(log_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_trial_metrics(log_path: Path, threshold_px: float = 20.0) -> dict:
    """Per-trial error metrics: RMSE, median, p90, std, pct within threshold, avg head size."""
    records = _read_jsonl(log_path)
    if not records:
        return {}

    errs = np.array(
        [
            math.hypot(r["pred_x_px"] - r["target_x_px"], r["pred_y_px"] - r["target_y_px"])
            for r in records
        ]
    )
    head_sizes = np.array([r.get("head_size_norm", 0.0) for r in records])

    return {
        "rmse": float(np.sqrt(np.mean(errs**2))),
        "median": float(np.median(errs)),
        "p90": float(np.percentile(errs, 90)),
        "std": float(np.std(errs)),
        "pct_within": float((errs < threshold_px).mean()),
        "avg_head_size": float(head_sizes.mean()),
        "n_frames": int(len(errs)),
    }


def aggregate_trials(log_paths: list[Path], threshold_px: float = 20.0) -> dict:
    """Aggregate per-trial metrics (e.g. a triplet of repeats) into mean/std per metric."""
    per_trial = [compute_trial_metrics(p, threshold_px=threshold_px) for p in log_paths]
    per_trial = [m for m in per_trial if m]
    if not per_trial:
        return {}

    agg: dict[str, float] = {}
    for key in ("rmse", "median", "p90", "std", "pct_within", "avg_head_size"):
        values = np.array([m[key] for m in per_trial])
        agg[f"{key}_mean"] = float(values.mean())
        agg[f"{key}_std"] = float(values.std())
    agg["n_trials"] = len(per_trial)
    return agg


class NaiveGazeToPixel:
    """Rough, uncalibrated pitch/yaw -> pixel mapping used for the `no_calibration` baseline.

    Assumes gaze angles within [-fov_rad, +fov_rad] linearly span the full screen, centered
    on the screen middle. This is a coarse approximation, not a fitted calibration.
    """

    def __init__(self, screen_width: int, screen_height: int, fov_rad: float = 0.35):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.fov_rad = fov_rad

    def gaze_to_pixel(self, pitch: float, yaw: float) -> tuple[int, int]:
        x = self.screen_width / 2 + (pitch / self.fov_rad) * (self.screen_width / 2)
        y = self.screen_height / 2 + (yaw / self.fov_rad) * (self.screen_height / 2)
        x = max(0, min(int(x), self.screen_width - 1))
        y = max(0, min(int(y), self.screen_height - 1))
        return x, y


def build_pixel_mapper(
    fit_type: str,
    screen_size: tuple[int, int],
    calibration_path: Path | None = None,
    ridge_alpha: float = 1.0,
    fov_rad: float = 0.35,
):
    """Return an object exposing `.gaze_to_pixel(pitch, yaw) -> (x, y)` for the given fit type."""
    screen_w, screen_h = screen_size
    if fit_type == "none":
        return NaiveGazeToPixel(screen_w, screen_h, fov_rad=fov_rad)
    if fit_type not in ("univariate", "bivariate"):
        raise ValueError(f"Unknown fit_type {fit_type!r}")
    if calibration_path is None:
        raise ValueError(f"--calibration is required when --fit-type={fit_type!r}")

    from virtual_cursor.calibration_loader import CalibrationLoader
    from virtual_cursor.gaze_to_pixel import GazeToPixel

    loader = CalibrationLoader(calibration_path, fit_type=fit_type, ridge_alpha=ridge_alpha)
    pitch_min, pitch_max, yaw_min, yaw_max = loader.get_gaze_bounds()
    poly_pitch_to_x, poly_yaw_to_y = loader.get_polynomials()
    coeffs_x, coeffs_y = loader.get_bivariate_coeffs()
    return GazeToPixel(
        screen_w,
        screen_h,
        poly_pitch_to_x=poly_pitch_to_x,
        poly_yaw_to_y=poly_yaw_to_y,
        coeffs_x=coeffs_x,
        coeffs_y=coeffs_y,
        fit_type=fit_type,
        pitch_bounds=(pitch_min, pitch_max),
        yaw_bounds=(yaw_min, yaw_max),
    )


def run_live_trial(
    session_id: str,
    trial_id: str,
    rep: int,
    condition: str,
    out_path: Path,
    gaze_capture,
    fit_type: str = "none",
    calibration_path: Path | None = None,
    ridge_alpha: float = 1.0,
    fov_rad: float = 0.35,
    rows: int = 3,
    cols: int = 3,
    dwell: float = 1.5,
    fps: int = 30,
    fullscreen: bool = True,
    windowed_size: tuple[int, int] = (1920, 1080),
    model_version: str = "l2cs",
) -> Path | None:
    """Display each static point in turn, recording real webcam/model predictions to JSONL.

    `gaze_capture` (a `calibration_tool.calibration.GazeCapture`, owned/cleaned up by the caller)
    supplies frames + gaze predictions; the pixel mapper is built once the real display size is
    known. Returns the output path, or None if aborted (ESC/window close) before any frame.
    """
    import cv2
    import pygame

    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}")

    pygame.init()
    recorder = None
    try:
        flags = pygame.FULLSCREEN if fullscreen else 0
        display = pygame.display.set_mode((0, 0) if fullscreen else windowed_size, flags)
        pygame.display.set_caption(f"Evaluation: {condition} rep{rep}")
        screen_size = display.get_size()

        points = generate_static_points(*screen_size, rows=rows, cols=cols)
        pixel_mapper = build_pixel_mapper(
            fit_type, screen_size, calibration_path, ridge_alpha, fov_rad
        )
        recorder = TrialRecorder(session_id, trial_id, rep, condition, screen_size, model_version)
        clock = pygame.time.Clock()
        aborted = False

        for point_index, (tx, ty) in enumerate(points):
            deadline = time.time() + dwell
            while time.time() < deadline and not aborted:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        aborted = True

                display.fill((255, 255, 255))
                pygame.draw.circle(display, (200, 0, 0), (int(tx), int(ty)), 20)
                pygame.display.flip()

                success, frame, gaze_result = gaze_capture.capture_frame()
                if success and gaze_result is not None and gaze_result.bboxes.size:
                    best = int(np.argmax(gaze_result.scores))
                    pitch, yaw = float(gaze_result.pitch[best]), float(gaze_result.yaw[best])
                    x1, y1, x2, y2 = gaze_result.bboxes[best][:4]
                    face_bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    brightness = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
                    recorder.add(
                        point_index,
                        (tx, ty),
                        pixel_mapper.gaze_to_pixel(pitch, yaw),
                        (pitch, yaw),
                        face_bbox,
                        brightness_mean=brightness,
                    )

                clock.tick(fps)
            if aborted:
                break
    finally:
        pygame.quit()

    if recorder is None or not recorder.records:
        return None
    return recorder.write(out_path)
