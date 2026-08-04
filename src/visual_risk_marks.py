"""High-precision detectors for OCR-invisible risk seals."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


_BIOHAZARD_REFERENCE = np.asarray(
    [
        [519, 136], [512, 142], [506, 159], [507, 165], [514, 168],
        [515, 180], [518, 185], [531, 187], [542, 197], [556, 202],
        [574, 201], [587, 192], [587, 188], [583, 186], [584, 166],
        [576, 164], [576, 155], [571, 147], [571, 138], [581, 137],
        [580, 132], [570, 128], [552, 129], [543, 133], [542, 140],
        [529, 140], [526, 136],
    ],
    dtype=np.int32,
).reshape(-1, 1, 2)


def _is_biohazard_contour(contour: np.ndarray) -> bool:
    return cv2.matchShapes(
        _BIOHAZARD_REFERENCE, contour, cv2.CONTOURS_MATCH_I1, 0
    ) <= 0.012


def purple_triangle_in_image(bgr: np.ndarray) -> bool:
    """Return whether a page contains the large purple triangular embargo seal."""
    if bgr.size == 0:
        return False
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    mask = (
        (hue >= 125)
        & (hue < 170)
        & (saturation > 20)
        & (value < 250)
    ).astype(np.uint8) * 255
    # Portraits and colored form headings live near the top.  The printed risk
    # seal is in the form body; ignoring the top band removes those distractors.
    mask[: int(0.30 * mask.shape[0]), :] = 0
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8)
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_area = float(mask.size)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area / page_area <= 0.0005:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        polygon = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(polygon) != 3:
            continue
        _x, _y, width, height = cv2.boundingRect(contour)
        if 0.65 <= min(width, height) / max(width, height) <= 1.0:
            return True
    return False


def red_biohazard_in_image(
    bgr: np.ndarray, *, include_occluded: bool = False
) -> bool:
    """Match the distinctive red circular biohazard adjudicator seal."""
    if bgr.size == 0:
        return False
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    mask = (
        ((hue < 14) | (hue >= 170))
        & (saturation > 28)
        & (value < 252)
    ).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((11, 11), dtype=np.uint8)
    )
    mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_area = float(mask.size)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not (0.00015 <= area / page_area <= 0.02):
            continue
        _x, _y, width, height = cv2.boundingRect(contour)
        perimeter = float(cv2.arcLength(contour, True))
        aspect = min(width, height) / max(width, height)
        circularity = 4.0 * np.pi * area / max(perimeter * perimeter, 1.0)
        if aspect < 0.62 or circularity < 0.10:
            continue
        if _is_biohazard_contour(contour):
            return True
    if not include_occluded:
        return False
    # Scan-band corruption can split the exact outer contour while preserving
    # a large, lower-right, topologically complex biohazard glyph.  This
    # secondary signature is deliberately about component geometry and holes,
    # not hue area alone (wax seals and stains are otherwise common).
    raw_mask = (
        ((hue < 14) | (hue >= 170))
        & (saturation > 28)
        & (value < 252)
    ).astype(np.uint8) * 255
    damaged = cv2.morphologyEx(
        raw_mask, cv2.MORPH_CLOSE, np.ones((11, 11), dtype=np.uint8)
    )
    damaged = cv2.dilate(
        damaged, np.ones((3, 3), dtype=np.uint8), iterations=1
    )
    damaged_contours, _ = cv2.findContours(
        damaged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    height_px, width_px = raw_mask.shape
    for contour in damaged_contours:
        area = float(cv2.contourArea(contour))
        area_fraction = area / float(raw_mask.size)
        if not 0.0045 <= area_fraction <= 0.007:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        aspect = min(width, height) / max(width, height)
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 4.0 * np.pi * area / max(perimeter * perimeter, 1.0)
        center_x = (x + width / 2.0) / width_px
        center_y = (y + height / 2.0) / height_px
        if not (
            0.65 <= aspect <= 0.85
            and 0.18 <= circularity <= 0.40
            and center_x > 0.72
            and center_y > 0.52
        ):
            continue
        pad = 5
        crop = raw_mask[
            max(0, y - pad) : min(height_px, y + height + pad),
            max(0, x - pad) : min(width_px, x + width + pad),
        ]
        _inner, hierarchy = cv2.findContours(
            crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        holes = (
            sum(1 for node in hierarchy[0] if node[3] >= 0)
            if hierarchy is not None
            else 0
        )
        if holes >= 8:
            return True
    return False


@lru_cache(maxsize=128)
def planetary_embargo_triangle(pdf_path: Path) -> bool:
    """Inspect rendered pages for the triangular planetary-embargo seal."""
    if not shutil.which("pdftoppm"):
        return False
    try:
        with tempfile.TemporaryDirectory(prefix="mib-risk-mark-") as tmp:
            prefix = Path(tmp) / "page"
            subprocess.run(
                ["pdftoppm", "-r", "144", "-png", str(pdf_path), str(prefix)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=8,
            )
            return any(
                purple_triangle_in_image(image)
                for path in sorted(Path(tmp).glob("page-*.png"))
                if (image := cv2.imread(str(path), cv2.IMREAD_COLOR)) is not None
            )
    except (OSError, subprocess.SubprocessError):
        return False


@lru_cache(maxsize=128)
def biohazard_red_seal(pdf_path: Path) -> bool:
    """Inspect low-resolution pages for the high-precision biohazard seal."""
    if not shutil.which("pdftoppm"):
        return False
    try:
        with tempfile.TemporaryDirectory(prefix="mib-red-risk-mark-") as tmp:
            prefix = Path(tmp) / "page"
            subprocess.run(
                ["pdftoppm", "-r", "72", "-jpeg", str(pdf_path), str(prefix)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=8,
            )
            return any(
                red_biohazard_in_image(image)
                for path in sorted(Path(tmp).glob("page-*.jpg"))
                if (image := cv2.imread(str(path), cv2.IMREAD_COLOR)) is not None
            )
    except (OSError, subprocess.SubprocessError):
        return False


@lru_cache(maxsize=128)
def occluded_biohazard_red_seal(pdf_path: Path) -> bool:
    """Inspect for the topology-preserving scan-band biohazard signature."""
    if not shutil.which("pdftoppm"):
        return False
    try:
        with tempfile.TemporaryDirectory(prefix="mib-occluded-red-mark-") as tmp:
            prefix = Path(tmp) / "page"
            subprocess.run(
                ["pdftoppm", "-r", "72", "-jpeg", str(pdf_path), str(prefix)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=8,
            )
            return any(
                red_biohazard_in_image(image, include_occluded=True)
                for path in sorted(Path(tmp).glob("page-*.jpg"))
                if (image := cv2.imread(str(path), cv2.IMREAD_COLOR)) is not None
            )
    except (OSError, subprocess.SubprocessError):
        return False
