"""Gated PDF-layer evidence for damaged raster-reconstructed packets."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RasterProfile:
    full_page_images: int = 0
    small_assets: int = 0
    max_dark_fraction: float = 0.0


@lru_cache(maxsize=128)
def raster_profile(pdf_path: Path) -> RasterProfile:
    """Count page rasters/assets and measure severe reconstructed-page ink."""
    try:
        from pypdf import PdfReader

        full_page_images = 0
        small_assets = 0
        max_dark_fraction = 0.0
        for page in PdfReader(str(pdf_path)).pages:
            for item in page.images:
                image = item.image
                width, height = image.size
                if width >= 1000 and height >= 1000:
                    full_page_images += 1
                    gray = np.asarray(image.convert("L"), dtype=np.uint8)
                    if gray.size:
                        max_dark_fraction = max(
                            max_dark_fraction, float((gray < 220).mean())
                        )
                elif 96 <= width <= 800 and 96 <= height <= 800:
                    small_assets += 1
        return RasterProfile(
            full_page_images=full_page_images,
            small_assets=small_assets,
            max_dark_fraction=max_dark_fraction,
        )
    except Exception:  # noqa: BLE001 - optional forensic evidence must abstain
        return RasterProfile()
