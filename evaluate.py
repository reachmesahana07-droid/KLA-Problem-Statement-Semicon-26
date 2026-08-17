#!/usr/bin/env python3
"""
Standalone evaluation script for the KLA grayscale image-restoration challenge.

Usage:
    python evaluate.py --input_dir ./test_images --output_dir ./outputs

The script expects a checkpoint compatible with RestorationNet.
Preferred checkpoint format:
{
    "model_state_dict": ...,
    "scale_factor": 2,
    "output_min": 0.0,
    "output_max": 1.0
}

Input:
    .npy files containing 2-D grayscale floating-point arrays.

Output:
    .npy files containing restored 2-D float32 arrays at 2x spatial resolution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Small residual block used by the restoration network."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class RestorationNet(nn.Module):
    """
    Lightweight fully-convolutional denoising + 2x super-resolution network.

    The network accepts either:
        B x 1 x 128 x 128
    or:
        B x 1 x 256 x 256

    and returns:
        B x 1 x 256 x 256
    or:
        B x 1 x 512 x 512
    respectively.

    A bicubic-upsampled input is used as a base image and the network learns
    a residual correction. This makes the model useful for both denoising
    and reconstruction of high-frequency details.
    """

    def __init__(self, base_channels: int = 48):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 3

        self.stem = nn.Conv2d(1, c1, 3, padding=1)

        self.enc1 = nn.Sequential(
            ResidualBlock(c1),
            ResidualBlock(c1),
        )

        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)

        self.enc2 = nn.Sequential(
            ResidualBlock(c2),
            ResidualBlock(c2),
        )

        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)

        self.bottleneck = nn.Sequential(
            ResidualBlock(c3),
            ResidualBlock(c3),
            ResidualBlock(c3),
        )

        self.up2_conv = nn.Conv2d(c3, c2, 3, padding=1)
        self.dec2 = nn.Sequential(
            ResidualBlock(c2),
            ResidualBlock(c2),
        )

        self.up1_conv = nn.Conv2d(c2, c1, 3, padding=1)
        self.dec1 = nn.Sequential(
            ResidualBlock(c1),
            ResidualBlock(c1),
        )

        self.residual_head = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, 1, 3, padding=1),
        )

    @staticmethod
    def _resize(x: torch.Tensor, size) -> torch.Tensor:
        return F.interpolate(
            x, size=size, mode="bilinear", align_corners=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_h, input_w = x.shape[-2:]
        target_size = (input_h * 2, input_w * 2)

        # Preserve the information in the original noisy input rather than
        # clipping it before the network sees it.
        base = F.interpolate(
            x, size=target_size, mode="bicubic", align_corners=False
        )

        s1 = self.enc1(self.stem(x))

        d1 = self.down1(s1)
        s2 = self.enc2(d1)

        d2 = self.down2(s2)
        b = self.bottleneck(d2)

        u2 = self._resize(b, s2.shape[-2:])
        u2 = self.up2_conv(u2)
        u2 = u2 + s2
        u2 = self.dec2(u2)

        u1 = self._resize(u2, s1.shape[-2:])
        u1 = self.up1_conv(u1)
        u1 = u1 + s1
        u1 = self.dec1(u1)

        residual = self.residual_head(u1)
        residual = self._resize(residual, target_size)

        return base + residual


# ---------------------------------------------------------------------------
# Checkpoint handling
# ---------------------------------------------------------------------------

def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, float, float]:
    """Load the trained model and output range from a checkpoint."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Place the trained checkpoint at checkpoints/best_model.pth "
            "or pass --checkpoint explicitly."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    output_min = 0.0
    output_max = 1.0

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        output_min = float(checkpoint.get("output_min", output_min))
        output_max = float(checkpoint.get("output_max", output_max))
    elif isinstance(checkpoint, dict):
        # Also accept a plain state dictionary.
        state_dict = checkpoint
    else:
        raise ValueError(
            "Unsupported checkpoint format. Save either a plain state_dict "
            "or a dictionary containing 'model_state_dict'."
        )

    model = RestorationNet()
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model, output_min, output_max


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_npy(path: Path) -> np.ndarray:
    """Load and validate one grayscale NumPy image."""

    array = np.load(path)

    if not isinstance(array, np.ndarray):
        raise ValueError(f"{path} did not contain a NumPy array.")

    if array.ndim != 2:
        raise ValueError(
            f"{path} must contain a 2-D grayscale array; "
            f"received shape {array.shape}."
        )

    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(
            f"{path} must contain numeric data; received {array.dtype}."
        )

    array = np.asarray(array, dtype=np.float32)

    if not np.isfinite(array).all():
        raise ValueError(f"{path} contains NaN or infinite values.")

    return array


@torch.inference_mode()
def restore_image(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    output_min: float,
    output_max: float,
) -> np.ndarray:
    """Run one image through the model and return a 2-D float32 result."""

    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(
        device=device, dtype=torch.float32
    )

    prediction = model(tensor)

    # The challenge permits noisy input values outside the ground-truth
    # range. Clipping is applied only after restoration.
    prediction = torch.clamp(prediction, output_min, output_max)

    result = prediction.squeeze(0).squeeze(0).cpu().numpy()
    return np.asarray(result, dtype=np.float32)


def evaluate(
    input_dir: Path,
    output_dir: Path,
    checkpoint_path: Path,
    device_name: str,
) -> None:
    """Evaluate every .npy image in input_dir."""

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available on this machine."
        )

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    model, output_min, output_max = load_checkpoint(
        checkpoint_path, device
    )

    image_paths = sorted(input_dir.glob("*.npy"))

    if not image_paths:
        raise FileNotFoundError(
            f"No .npy files found in {input_dir}"
        )

    print(
        f"Found {len(image_paths)} images. "
        f"Output range: [{output_min}, {output_max}]"
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    for index, image_path in enumerate(image_paths, start=1):
        image = load_npy(image_path)

        restored = restore_image(
            model=model,
            image=image,
            device=device,
            output_min=output_min,
            output_max=output_max,
        )

        output_path = output_dir / image_path.name
        np.save(output_path, restored)

        print(
            f"[{index:04d}/{len(image_paths):04d}] "
            f"{image_path.name}: "
            f"{tuple(image.shape)} -> {tuple(restored.shape)}"
        )

    print("Evaluation complete.")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the trained KLA grayscale image-restoration model "
            "on every .npy image in a directory."
        )
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        type=Path,
        help="Directory containing degraded .npy test images.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory where restored .npy files will be written.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_model.pth"),
        help=(
            "Path to the trained model checkpoint. "
            "Default: checkpoints/best_model.pth"
        ),
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. Default: auto.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        evaluate(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            device_name=args.device,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
