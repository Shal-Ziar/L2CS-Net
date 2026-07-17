"""Pseudo-label generation and model fine-tuning for personalised gaze.

Two public functions:

``generate_pseudo_labels``
    Loads the *original* (un-fine-tuned) L2CS model, runs inference on all
    face crops saved by the collection step, and writes a ``labels.csv`` into
    the session directory.

``run_finetune``
    Loads a base model, builds a :class:`~personal_trainer.dataset.PersonalDataset`
    from the session directory, runs a short training loop with the same
    hybrid-loss / 3-group-Adam strategy as ``train.py``, and saves the
    fine-tuned weights as a dated ``.safetensors`` file.
"""

import csv
import cv2
import numpy as np
import torch
import torch.nn as nn
from datetime import date
from l2cs import getArch, select_device
from l2cs.utils import prep_input_numpy
from pathlib import Path
from personal_trainer.dataset import PersonalDataset
from safetensors.torch import load_file, save_file
from torch.autograd import Variable
from torch.utils.data import DataLoader
from typing import Generator

# ---------------------------------------------------------------------------
# Parameter-group helpers — identical to train.py so fine-tuning behaviour
# matches the original training scheme.
# ---------------------------------------------------------------------------


def _get_ignored_params(model: nn.Module) -> Generator:
    b = [model.conv1, model.bn1, model.fc_finetune]
    for block in b:
        for module_name, module in block.named_modules():
            if "bn" in module_name:
                module.eval()
            yield from module.parameters()


def _get_non_ignored_params(model: nn.Module) -> Generator:
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for block in b:
        for module_name, module in block.named_modules():
            if "bn" in module_name:
                module.eval()
            yield from module.parameters()


def _get_fc_params(model: nn.Module) -> Generator:
    b = [model.fc_yaw_gaze, model.fc_pitch_gaze]
    for block in b:
        yield from block.parameters()


# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------


def _load_model(snapshot: Path, arch: str, device: torch.device) -> nn.Module:
    model = getArch(arch, 90)
    if snapshot.suffix == ".safetensors":
        model.load_state_dict(load_file(snapshot, device=str(device)))
    else:
        model.load_state_dict(torch.load(snapshot, map_location=device))
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_pseudo_labels(
    data_dir: Path,
    snapshot: Path,
    arch: str = "ResNet50",
    gpu: str = "0",
    batch_size: int = 32,
) -> Path:
    """Run the base L2CS model on all collected crops and write ``labels.csv``.

    Uses the *original* weights so that the labels are not contaminated by any
    previous fine-tuning pass.

    Args:
        data_dir:   Session directory containing ``metadata.csv`` and crops.
        snapshot:   Path to base model weights (.pkl or .safetensors).
        arch:       Model architecture (must match ``snapshot``).
        gpu:        GPU device id ("0", "1", …) or "cpu".
        batch_size: Number of crops processed per forward pass.

    Returns:
        Path to the written ``labels.csv``.
    """
    device = select_device(gpu)
    model = _load_model(snapshot, arch, device)
    model.to(device)
    model.eval()

    softmax = nn.Softmax(dim=1).to(device)
    idx_tensor = torch.FloatTensor(list(range(90))).to(device)

    metadata_path = data_dir / "metadata.csv"
    with open(metadata_path, newline="") as f:
        rows = list(csv.DictReader(f))

    labels_path = data_dir / "labels.csv"
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "pitch_deg", "yaw_deg"])
        writer.writeheader()

        with torch.no_grad():
            for i in range(0, len(rows), batch_size):
                batch_rows = rows[i : i + batch_size]
                imgs = []
                for row in batch_rows:
                    img_bgr = cv2.imread(str(data_dir / row["image_path"]))
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    imgs.append(img_rgb)

                # prep_input_numpy handles (N, H, W, C) → (N, 3, 448, 448) tensor
                img_tensor = prep_input_numpy(np.stack(imgs), device)

                gaze_pitch, gaze_yaw = model(img_tensor)
                pitch_pred = torch.sum(softmax(gaze_pitch) * idx_tensor, dim=1) * 4 - 180
                yaw_pred = torch.sum(softmax(gaze_yaw) * idx_tensor, dim=1) * 4 - 180

                for j, row in enumerate(batch_rows):
                    writer.writerow(
                        {
                            "image_path": row["image_path"],
                            "pitch_deg": float(pitch_pred[j].item()),
                            "yaw_deg": float(yaw_pred[j].item()),
                        }
                    )

    print(f"Pseudo-labels written: {labels_path}  ({len(rows)} samples)")
    return labels_path


def run_finetune(
    data_dir: Path,
    snapshot: Path,
    arch: str = "ResNet50",
    num_epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-5,
    alpha: float = 1.0,
    output_dir: Path = Path("output/personal"),
    gpu: str = "0",
) -> Path:
    """Fine-tune a base L2CS model on the personalised dataset.

    Follows the same hybrid-loss, 3-group Adam strategy as ``train.py``.
    Saves the result as ``personal_{arch}_{YYYY-MM-DD}.safetensors``.

    Args:
        data_dir:   Session directory with ``labels.csv`` and face crops.
        snapshot:   Base model weights (.pkl or .safetensors).
        arch:       Architecture name (must match snapshot).
        num_epochs: Training epochs.
        batch_size: Mini-batch size.
        lr:         Learning rate for non-frozen layers.
        alpha:      Weight of MSE regression loss (same as train.py ``--alpha``).
        output_dir: Directory to write the fine-tuned model.
        gpu:        GPU device id or "cpu".

    Returns:
        Path to the saved ``.safetensors`` file.
    """
    device = select_device(gpu)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(snapshot, arch, device)
    model.to(device)

    dataset = PersonalDataset(data_dir)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss().to(device)
    reg_criterion = nn.MSELoss().to(device)
    softmax = nn.Softmax(dim=1).to(device)
    idx_tensor = Variable(torch.FloatTensor(list(range(90)))).to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": _get_ignored_params(model), "lr": 0},
            {"params": _get_non_ignored_params(model), "lr": lr},
            {"params": _get_fc_params(model), "lr": lr},
        ],
        lr,
    )

    # Keep ALL BatchNorm layers in eval mode (running statistics) so that the
    # model's forward pass is identical to the one used in generate_pseudo_labels,
    # which calls model.eval().  If any BN layer uses batch statistics here
    # (batch of 8 user images vs Gaze360 running stats), predictions diverge
    # from the pseudo-labels and epoch-1 MSE loss explodes — especially for
    # pitch.  train/eval mode does NOT affect gradient flow or requires_grad.
    model.eval()

    print(
        f"Fine-tuning {arch} for {num_epochs} epoch(s) on {len(dataset)} samples "
        f"(batch={batch_size}, lr={lr})"
    )

    for epoch in range(num_epochs):
        sum_pitch = sum_yaw = n_iter = 0

        for images, labels, cont_labels in loader:
            images = Variable(images).to(device)
            label_pitch = Variable(labels[:, 0]).to(device)
            label_yaw = Variable(labels[:, 1]).to(device)
            label_pitch_cont = Variable(cont_labels[:, 0]).to(device)
            label_yaw_cont = Variable(cont_labels[:, 1]).to(device)

            pitch, yaw = model(images)

            loss_pitch = criterion(pitch, label_pitch)
            loss_yaw = criterion(yaw, label_yaw)

            pitch_pred = torch.sum(softmax(pitch) * idx_tensor, 1) * 4 - 180
            yaw_pred = torch.sum(softmax(yaw) * idx_tensor, 1) * 4 - 180

            loss_pitch = loss_pitch + alpha * reg_criterion(pitch_pred, label_pitch_cont)
            loss_yaw = loss_yaw + alpha * reg_criterion(yaw_pred, label_yaw_cont)

            sum_pitch += loss_pitch.item()
            sum_yaw += loss_yaw.item()
            n_iter += 1

            optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward(
                [loss_pitch, loss_yaw],
                [torch.tensor(1.0).to(device), torch.tensor(1.0).to(device)],
            )
            optimizer.step()

        print(
            f"  Epoch [{epoch + 1}/{num_epochs}]  "
            f"pitch_loss={sum_pitch / n_iter:.4f}  yaw_loss={sum_yaw / n_iter:.4f}"
        )

    today = date.today().strftime("%Y-%m-%d")
    save_path = output_dir / f"personal_{arch}_{today}.safetensors"
    save_file(model.state_dict(), str(save_path))
    print(f"Model saved: {save_path}")
    return save_path
