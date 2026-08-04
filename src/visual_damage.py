"""Cheap visual recovery for OCR-invisible registry portrait obstruction."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import cv2
from PIL import Image


def registry_crop_is_obscured(image: Image.Image) -> bool:
    """Detect the saturated overlay used to obscure a registry portrait."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    crop = rgb[
        int(0.40 * height) : int(0.66 * height),
        int(0.70 * width) : int(0.94 * width),
    ]
    if not crop.size:
        return False
    channel_range = crop.max(axis=2).astype(np.int16) - crop.min(axis=2)
    gray = crop.mean(axis=2)
    saturated_fraction = float((channel_range > 12).mean())
    dark_fraction = float((gray < 220).mean())
    saturated_mask = (channel_range > 12).astype(np.uint8)
    _count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        saturated_mask
    )
    largest_component = (
        float(stats[1:, cv2.CC_STAT_AREA].max()) / saturated_mask.size
        if len(stats) > 1
        else 0.0
    )
    return (
        saturated_fraction > 0.20
        and dark_fraction > 0.08
        and largest_component > 0.25
    )


@lru_cache(maxsize=256)
def registry_portrait_obscured(pdf_path: Path, page_number: int) -> bool:
    """Render only a native-text registry page and inspect its portrait ROI."""
    if not shutil.which("pdftoppm"):
        return False
    if not 1 <= page_number <= 8:
        return False

    try:
        with tempfile.TemporaryDirectory(prefix="mib-registry-damage-") as tmp:
            prefix = Path(tmp) / "page"
            subprocess.run(
                [
                    "pdftoppm", "-f", str(page_number), "-l", str(page_number),
                    "-r", "72", "-png", "-singlefile", str(pdf_path), str(prefix),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=4,
            )
            with Image.open(prefix.with_suffix(".png")) as image:
                return registry_crop_is_obscured(image)
    except (OSError, subprocess.SubprocessError):
        return False
