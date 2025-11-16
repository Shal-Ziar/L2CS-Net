import cv2
import logging
import pathlib
import time
import torch
import torch.backends.cudnn as cudnn
from l2cs import Pipeline, select_device

CWD = pathlib.Path.cwd()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Called with args:")

    cudnn.enabled = True
    arch = "ResNet50"
    cam = 0
    # snapshot_path = args.snapshot

    gaze_pipeline = Pipeline(
        weights=CWD / "models" / "L2CSNet_gaze360.pkl",
        arch="ResNet50",
        device=select_device("gpu:0", batch_size=1),
    )
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    with torch.no_grad():
        while True:

            # Get frame
            success, frame = cap.read()
            start_fps = time.time()

            if not success:
                print("Failed to obtain frame")
                time.sleep(0.1)

            # Process frame
            results = gaze_pipeline.step(frame)
            print(results)
