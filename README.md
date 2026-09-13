


 <p align="center">
  <img src="https://github.com/Ahmednull/Storage/blob/main/gaze.gif" alt="animated" />
</p>


___

# L2CS-Net

The official PyTorch implementation of L2CS-Net for gaze estimation and tracking.

## Installation
<img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" />

Install package with the following:

```
pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main
```

Or, you can git clone the repo and install with the following:

```
pip install [-e] .
```

Now you should be able to import the package with the following command:

```
$ python
>>> import l2cs
```

## Usage

Detect face and predict gaze from webcam

```python
from l2cs import Pipeline, render
import cv2

gaze_pipeline = Pipeline(
    weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
    arch='ResNet50',
    device=torch.device('cpu') # or 'gpu'
)

cap = cv2.VideoCapture(cam)
_, frame = cap.read()

# Process frame and visualize
results = gaze_pipeline.step(frame)
frame = render(frame, results)
```

## Demo
* Download the pre-trained models from [here](https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd?usp=sharing) and Store it to *models/*.
*  Run:
```
 python demo.py \
 --snapshot models/L2CSNet_gaze360.pkl \
 --gpu 0 \
 --cam 0 \
```
This means the demo will run using *L2CSNet_gaze360.pkl* pretrained model

## Community Contributions

- [Gaze Detection and Eye Tracking: A How-To Guide](https://blog.roboflow.com/gaze-direction-position/): Use L2CS-Net through a HTTP interface with the open source Roboflow Inference project.

## MPIIGaze
We provide the code for train and test MPIIGaze dataset with leave-one-person-out evaluation.

### Prepare datasets
* Download **MPIIFaceGaze dataset** from [here](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation).
* Apply data preprocessing from [here](http://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/).
* Store the dataset to *datasets/MPIIFaceGaze*.

### Train
```
 python train.py \
 --dataset mpiigaze \
 --snapshot output/snapshots \
 --gpu 0 \
 --num_epochs 50 \
 --batch_size 16 \
 --lr 0.00001 \
 --alpha 1 \

```
This means the code will perform leave-one-person-out training automatically and store the models to *output/snapshots*.

### Test
```
 python test.py \
 --dataset mpiigaze \
 --snapshot output/snapshots/snapshot_folder \
 --evalpath evaluation/L2CS-mpiigaze  \
 --gpu 0 \
```
This means the code will perform leave-one-person-out testing automatically and store the results to *evaluation/L2CS-mpiigaze*.

To get the average leave-one-person-out accuracy use:
```
 python leave_one_out_eval.py \
 --evalpath evaluation/L2CS-mpiigaze  \
 --respath evaluation/L2CS-mpiigaze  \
```
This means the code will take the evaluation path and outputs the leave-one-out gaze accuracy to the *evaluation/L2CS-mpiigaze*.

## Gaze360
We provide the code for train and test Gaze360 dataset with train-val-test evaluation.

### Prepare datasets
* Download **Gaze360 dataset** from [here](http://gaze360.csail.mit.edu/download.php).

* Apply data preprocessing from [here](http://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/).

* Store the dataset to *datasets/Gaze360*.


### Train
```
 python train.py \
 --dataset gaze360 \
 --snapshot output/snapshots \
 --gpu 0 \
 --num_epochs 50 \
 --batch_size 16 \
 --lr 0.00001 \
 --alpha 1 \

```
This means the code will perform training and store the models to *output/snapshots*.

### Test
```
 python test.py \
 --dataset gaze360 \
 --snapshot output/snapshots/snapshot_folder \
 --evalpath evaluation/L2CS-gaze360  \
 --gpu 0 \
```
This means the code will perform testing on snapshot_folder and store the results to *evaluation/L2CS-gaze360*.

## CLI Usage

The repository provides several command-line entry points for common workflows. Below are the most-used commands and example invocations; use `-h` on any command to list all flags.

- **Demo (live webcam preview)**: Runs a real-time demo with a model.

```
python demo.py --snapshot models/L2CSNet_gaze360.pkl --device gpu:0 --cam 0
```

- **Train (train models)**: Train on a dataset (Gaze360 or MPIIGaze).

```
python train.py --dataset gaze360 --snapshot output/snapshots --gpu 0 --num_epochs 50 --batch_size 16 --lr 1e-5
```

- **Test / Evaluate**: Run evaluation over saved snapshots.

```
python test.py --dataset gaze360 --snapshot output/snapshots/snapshot_folder --evalpath evaluation/L2CS-gaze360 --gpu 0
```

- **Calibration Tool**: Full-screen calibration UI or background daemon. Use the module entrypoint or the package CLI.

Run full 9-point calibration (interactive):

```
python -m calibration_tool calibrate --model models/L2CSNet_gaze360.pkl --device gpu:0 --camera 0
```

Start background micro-calibration daemon (press `c` to collect 5-point checkpoints):

```
python -m calibration_tool daemon --model models/L2CSNet_gaze360.pkl --device gpu:0 --camera 0
```

- **Personal Trainer**: Capture personal face crops and fine-tune a model.

Capture face crops during calibration:

```
python -m personal_trainer collect --output-dir training_data --windowed
```

Fine-tune using a collected session:

```
python -m personal_trainer finetune --data-dir training_data/<session-id> --snapshot models/l2cs_gaze360_resnet50.safetensors --epochs 10 --batch-size 8 --lr 1e-5
```

Full pipeline (collect + finetune):

```
python -m personal_trainer run --snapshot models/l2cs_gaze360_resnet50.safetensors
```

- **Virtual Cursor**: Run the interactive cursor using a calibration file.

```
python -m virtual_cursor --calibration calibration/calibration_YYYY-MM-DD.jsonl --model models/l2cs_gaze360_resnet50.safetensors --device cuda --cam 0 --fullscreen --trial
```

Notes:
- Use `-h` or `--help` with any script or module (for example `python demo.py -h`) to see all flags and defaults.
- Device flags accept values like `cpu`, `gpu:0`, or `cuda` depending on the entry point.
- Calibration files are JSONL files written by the calibration tool and stored under `calibration_tool/calibration_data/` by default.

## Known Issues
 - Calibration for individuals isn't ideal. Model has trouble with tilted heads
 - Personalised calibration improves results somewhat. 5 epochs tried
 - Need to calculate some error metric to show results.

## Evaluation module

A compact evaluation toolkit is available under `accuracy_evaluation/`:
`accuracy_evaluation/evaluate.py` is the reusable library (static-point grids, JSONL
trial recording, pixel mapping, metrics) and `accuracy_evaluation/cli.py` wraps it in
a CLI with three subcommands: `mock` (simulated data, no camera/model needed), `live`
(real webcam + L2CS model measurements), and `aggregate` (combine triplet repeats
into mean/std metrics). Run `python3 -m accuracy_evaluation -h` or `... <subcommand>
-h` for the full flag list.

### `mock` — simulated dry-run

Writes `accuracy_evaluation_output/<session>_<trial>_<condition>_rep<n>.jsonl` using a
simulated predictor (useful for testing the pipeline/metrics without hardware):

```bash
python3 -m accuracy_evaluation mock --condition no_calibration --rows 3 --cols 3 --seed 42
```

Common flags shared by `mock` and `live`: `--session-id`, `--trial-id`, `--rep`,
`--condition` (one of `no_calibration`, `polynomial_univariate`,
`polynomial_bivariate`, `post_training`, `ring_light`), `--rows`/`--cols`,
`--dwell`, `--fps`, `--out-dir`, `--threshold-px`. `mock` additionally takes
`--screen-w`/`--screen-h` and `--seed` (reuse the same seed to reproduce a run).

### `live` — real camera + model measurements

Opens a fullscreen (or windowed) pygame target display, captures webcam frames
through the L2CS pipeline (`calibration_tool.calibration.GazeCapture`), maps
each predicted gaze angle to a screen pixel, and records genuine per-frame error
data to JSONL. Press `Esc` to abort a trial early.

```bash
python3 -m accuracy_evaluation live --condition no_calibration --fit-type none
```

Live-only flags:
- `--model` — model weights path (default `models/l2cs_gaze360_resnet50.safetensors`)
- `--device` — `cpu`, `gpu:0`, etc. (default `cpu`)
- `--camera` — camera index (default `0`)
- `--fit-type {none,univariate,bivariate}` — how gaze angles are mapped to pixels:
  - `none` — rough uncalibrated linear estimate (`--fov-rad`, default `0.35`)
  - `univariate`/`bivariate` — fitted polynomial from an existing calibration file
    (requires `--calibration`)
- `--calibration` — path to a calibration JSONL (required for `univariate`/`bivariate`)
- `--ridge-alpha` — ridge regression regularization used when loading a bivariate calibration
- `--windowed` — run in a window instead of fullscreen (`--screen-w`/`--screen-h` as fallback size)
- `--model-version` — free-text tag recorded in each JSONL row

Example invocations per condition:

```bash
# No calibration baseline — naive linear gaze-to-pixel mapping
python3 -m accuracy_evaluation live --condition no_calibration --fit-type none

# Polynomial calibration (univariate / bivariate) using an existing calibration file
python3 -m accuracy_evaluation live --condition polynomial_univariate --fit-type univariate \
  --calibration calibration_tool/calibration_data/calibration_2026-07-17_15-24-40.jsonl

python3 -m accuracy_evaluation live --condition polynomial_bivariate --fit-type bivariate \
  --calibration calibration_tool/calibration_data/calibration_2026-07-17_15-24-40.jsonl

# Post personal-training model weights
python3 -m accuracy_evaluation live --condition post_training \
  --model output/personal/personal_ResNet50_2026-07-17.safetensors \
  --fit-type univariate --calibration calibration_tool/calibration_data/calibration_2026-07-17_15-24-40.jsonl

# Ring light condition — same mapping as above, run under direct ring-light illumination
python3 -m accuracy_evaluation live --condition ring_light --fit-type bivariate \
  --calibration calibration_tool/calibration_data/calibration_2026-07-17_15-24-40.jsonl
```

`live` requires `opencv-python`, `pygame`, `torch`, and `batch_face` (already
project dependencies) plus an accessible webcam.

### JSONL schema

Per-frame fields written by both `mock` and `live`:
- `session_id`, `trial_id`, `rep`, `condition`, `timestamp`, `frame_idx`, `point_index`
- `target_x_px`, `target_y_px`, `pred_x_px`, `pred_y_px` (pixel coords)
- `pred_pitch`, `pred_yaw`
- `face_bbox` (object with `x,y,w,h`), `face_area`, `face_height_px`, `face_size_norm`
- `head_size_px`, `head_size_norm` — relative head-size proxy (sqrt(area) normalized by screen diagonal)
- `brightness_mean`, `camera_meta`, `model_version`, `extras`

### `aggregate` — combine triplet repeats

```bash
python3 -m accuracy_evaluation aggregate "*_no_calibration_rep*.jsonl" --out-dir accuracy_evaluation_output
```

Or from Python:

```python
from pathlib import Path
from accuracy_evaluation.evaluate import aggregate_trials

paths = sorted(Path("accuracy_evaluation_output").glob("*_no_calibration_rep*.jsonl"))
print(aggregate_trials(paths))
```
