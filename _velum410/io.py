"""Lossless RGB pixel I/O. ICC metadata is part of a color's definition."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .color import RGBEncoding


@dataclass
class RGBImage:
    rgb: np.ndarray
    encoding: RGBEncoding


def load(path: str | Path) -> RGBImage:
    path = Path(path)
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as data:
            if str(data['encoding']) != 'linear-srgb-float64':
                raise ValueError('Unsupported scientific RGB encoding.')
            rgb = np.array(data['rgb'], dtype=np.float64)
        encoding = RGBEncoding.linear_srgb()
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError('Expected H,W,3 float64 RGB.')
        encoding.linear(rgb.reshape(-1,3))
        return RGBImage(rgb, encoding)
    with Image.open(path) as image:
        image.load()
        profile = image.info.get('icc_profile')
        # Non-opaque pixels require a specified background/alpha measure. Never
        # silently composite: it would invent colors and alter the distribution.
        if image.mode not in ('RGB', 'RGBA', 'L', 'LA', 'P'):
            raise ValueError(f'{path}: only 8-bit RGB/grayscale/indexed inputs are supported.')
        if 'A' in image.getbands() or 'transparency' in image.info:
            if np.any(np.asarray(image.convert('RGBA'))[:, :, 3] != 255):
                raise ValueError(f'{path}: non-opaque alpha needs an explicit compositing policy.')
        if not profile and image.info.get('gamma') is not None and 'srgb' not in image.info:
            raise ValueError(f'{path}: PNG gAMA without sRGB/ICC is ambiguous; supply an RGB ICC profile.')
        if not profile and image.info.get('chromaticity') is not None and 'srgb' not in image.info:
            raise ValueError(f'{path}: PNG chromaticities without sRGB/ICC are unsupported.')
        encoding = RGBEncoding.from_icc(profile)
        # Orientation corrects coordinates without changing any pixel colors.
        oriented = ImageOps.exif_transpose(image)
        rgb = np.array(oriented.convert('RGB'), dtype=np.uint8)
    return RGBImage(rgb, encoding)


def save(path: str | Path, rgb: np.ndarray, encoding: RGBEncoding) -> None:
    path = Path(path)
    if path.suffix.lower() != '.png':
        raise ValueError('Output must be lossless PNG; JPEG would violate palette membership.')
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, icc_profile=encoding.icc)
