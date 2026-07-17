"""PyTorch dataset for personalised gaze fine-tuning.

Reads a ``labels.csv`` produced by :mod:`personal_trainer.finetune` and
serves (image, binned_labels, cont_labels) triples compatible with the
training loop in :mod:`personal_trainer.finetune`.

The image transforms and bin scheme mirror Gaze360 in ``l2cs/datasets.py``
so that the model head dimensions stay consistent:

- 90 bins, 4 °/bin, range [−180 °, 180 °)
- Resize 448 → ToTensor → ImageNet normalise
"""

import csv
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from typing import Tuple

# Matches Gaze360 transforms (l2cs/datasets.py + train.py)
_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_BINS = np.array(range(-180, 180, 4))  # 90 left-edges: −180, −176, … , 176
_NUM_BINS = 90


class PersonalDataset(Dataset):
    """Face-crop dataset with pseudo-labels for one user session."""

    def __init__(self, data_dir: Path):
        """
        Args:
            data_dir: Session directory that contains ``labels.csv`` and the
                      face-crop sub-directories written by
                      :class:`~personal_trainer.collect.ImageCollectionSession`.
        """
        self.data_dir = data_dir
        self.samples: list = []

        labels_path = data_dir / "labels.csv"
        with open(labels_path, newline="") as f:
            for row in csv.DictReader(f):
                self.samples.append(
                    {
                        "image_path": data_dir / row["image_path"],
                        "pitch_deg": float(row["pitch_deg"]),
                        "yaw_deg": float(row["yaw_deg"]),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor]:
        """Return (image, binned_labels, cont_labels).

        - ``image``: float32 tensor (3, 448, 448)
        - ``binned_labels``: int64 numpy array [pitch_bin, yaw_bin]
        - ``cont_labels``: float32 tensor [pitch_deg, yaw_deg]
        """
        sample = self.samples[idx]

        img = Image.open(sample["image_path"]).convert("RGB")
        img = _TRANSFORMS(img)

        pitch_deg = sample["pitch_deg"]
        yaw_deg = sample["yaw_deg"]

        binned = np.digitize([pitch_deg, yaw_deg], _BINS) - 1
        binned = np.clip(binned, 0, _NUM_BINS - 1).astype(np.int64)

        cont_labels = torch.FloatTensor([pitch_deg, yaw_deg])
        return img, binned, cont_labels
