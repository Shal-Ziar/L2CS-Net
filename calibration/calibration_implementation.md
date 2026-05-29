# Calibration Tool Implementation Plan

## Overview
Interactive calibration tool for L2CS gaze detection. Captures gaze data at 9 fixed grid points using a fullscreen pygame interface. User presses spacebar to cycle through: select point (green) → record 2 seconds (blue) → done (grey). Gaze frames captured continuously, stored per-point as `CalibrationPoint` objects in timestamped JSONL file.

## Implementation Steps

### Phase 1: Calibration Interface & State Management
1. Create `CalibrationInterface` class in `calibration/calibration.py`
   - Initialize pygame fullscreen with system display resolution
   - Define 3×3 grid layout (9 points), calculating center positions based on screen dimensions (margins ~10% from edges)
   - Manage point state machine: RED (inactive) → GREEN (selected) → BLUE (recording) → GREY (done)
   - Render points as circles + numbers (1-9), draw crosshair at center
   - Handle spacebar and ESC key events
   - Calculate point coordinates and auto-scale rendering for different screen sizes

2. Define calibration state enum in `calibration/types.py` if needed
   - States: `INACTIVE`, `SELECTED`, `RECORDING`, `COMPLETED`

### Phase 2: Gaze Pipeline Integration
3. Integrate gaze model in `CalibrationInterface` or separate `GazeCapture` class
   - Initialize L2CS Pipeline (from `vector_output.py` pattern) with `select_device("gpu:0")` and model path
   - Set up webcam frame capture loop using OpenCV
   - Process frames at all times (30+ fps if available)
   - Handle case where no face detected: skip recording for that point (leave grey), continue to next point on spacebar

4. Frame preprocessing & gaze extraction
   - Resize/normalize frames for L2CS model input
   - Extract `GazeResultContainer` (pitch, yaw, landmarks, confidence)
   - Queue frames for recording only during RECORDING state

### Phase 3: Data Capture & Storage
5. Recording logic
   - On spacebar (first press): Current point → GREEN, no recording yet
   - On spacebar (second press): Current point → BLUE, **start 2-second timer**, collect all gaze frames for this point
   - After 2 seconds: Current point → GREY, finalize `CalibrationPoint` object
   - Store `CalibrationPoint` with `calibration_point` ID (1-9) and list of `GazeResultContainer` objects

6. Metadata collection & session file creation
   - Capture system info: screen resolution, timestamp (ISO format), auto-detect camera/device info
   - Generate session filename: `calibration/callibration_data/calibration_{YYYY-MM-DD_HH-MM-SS}.jsonl`
   - Each line in JSONL: serialized `CalibrationPoint` (Pydantic `model_dump_json()`)
   - Add header line with metadata: `{timestamp, user_id (optional), screen_width, screen_height, device_info}`

### Phase 4: Post-Calibration Visualization
7. Visualization function in new `calibration/visualize.py`
   - Parse completed JSONL file
   - For each calibration point: extract mean/std pitch/yaw, convert to screen coordinates
   - Generate scatter plot: overlay gaze points on expected 3×3 grid locations
   - Generate heatmap: 2D histogram of gaze distribution across screen
   - Save plots to `calibration/callibration_data/{session_id}_scatter.png` and `{session_id}_heatmap.png`

### Phase 5: Main Entry Point
8. Create `calibration/cli.py` or extend `calibration/calibration.py`
   - CLI entry point: `python -m calibration.cli [--user-id USER] [--output-dir DIR]`
   - Initialize `CalibrationInterface`, start main loop
   - On completion or ESC: save JSONL file, trigger visualization, print summary

## Key Specifications

- **Interface**: Fullscreen pygame, auto-detect system resolution
- **Points**: 3×3 grid layout, ~10% margin from edges, dynamically calculated
- **Frame capture**: Continuous (all frames), no subsampling
- **Recording window**: 2 seconds per point, silent (no visual timer)
- **No-face handling**: Skip recording, point stays grey, user can recalibrate
- **Output format**: JSONL (one `CalibrationPoint` per line, Pydantic serialization)
- **State colors**: RED (inactive) → GREEN (selected) → BLUE (recording) → GREY (done)
- **Exit**: ESC at any time (clean abort, no partial save)

## Files to Create/Modify

- **`calibration/calibration.py`** — `CalibrationInterface` class, pygame setup, point rendering, state machine, spacebar handler, frame capture loop
- **`calibration/types.py`** — Add `PointState` enum; verify `CalibrationPoint` and `GazeResultContainer`
- **`calibration/visualize.py`** (new) — Post-calibration scatter plot and heatmap generation; JSONL parsing
- **`calibration/cli.py`** (new) — Entry point with argument parsing
- **`calibration/callibration_data/`** — Directory exists for JSONL and PNG output

## Verification Checklist

- [ ] Pygame window opens fullscreen, 9 points render in 3×3 grid with correct numbering
- [ ] Spacebar cycles: RED → GREEN → BLUE (2 sec) → GREY
- [ ] No-face detection: point stays grey when face not in frame
- [ ] JSONL file created with timestamped filename
- [ ] Each line parses as valid `CalibrationPoint` JSON
- [ ] Gaze data (pitch, yaw) within expected range (-π to π)
- [ ] Scatter plot and heatmap PNGs generated post-completion
- [ ] ESC exits cleanly
- [ ] Metadata (timestamp, screen resolution, camera info) included in JSONL header
