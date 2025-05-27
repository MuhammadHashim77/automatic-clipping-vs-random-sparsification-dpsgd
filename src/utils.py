"""
Reusable helpers for the Pneumothorax-DP project
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

# ---------- DICOM ► uint8 -------------------------------------------------- #
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut


def dcm_to_uint8(path: str | Path, *, voi_lut: bool = True) -> np.ndarray:
    """
    Read a DICOM file and return an 8-bit numpy array in the range [0, 255].

    Parameters
    ----------
    path : str | Path
        File path to the .dcm file.
    voi_lut : bool, default True
        If True, apply the VOI LUT or Window Center/Width contained
        in the DICOM tags (recommended for chest X-ray display).

    Handles • rescale slope/intercept
            • VOI LUT / window level
            • MONOCHROME1 inversion
    """
    ds = pydicom.dcmread(str(path))

    # 1) Raw pixel data → float32
    img = ds.pixel_array.astype(np.float32)

    # 2) Modality LUT (rescale slope & intercept)
    img = apply_modality_lut(img, ds)

    # 3) VOI LUT (windowing)
    if voi_lut:
        img = apply_voi_lut(img, ds)

    # 4) Normalise to 0-255
    img -= img.min()
    img /= max(img.max(), 1e-3)
    img *= 255.0

    # 5) Invert MONOCHROME1
    if ds.PhotometricInterpretation == "MONOCHROME1":
        img = 255.0 - img

    return img.astype(np.uint8)  # shape (H, W)

# ---------- RLE ► mask ----------------------------------------------------- #


def rle_decode(rle: str, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert the run-length-encoded string from train-rle.csv
    into a binary mask of shape (H, W).

    Parameters
    ----------
    rle : str
        The run-length string, e.g. "3 4 10 5 …".
        "-1" means "no mask" (empty array of zeros).
    shape : (H, W)
        Height and width of the target image.

    Returns
    -------
    mask : np.ndarray, dtype uint8, values {0,1}
    """
    if rle == "-1":
        return np.zeros(shape, dtype=np.uint8)

    s = list(map(int, rle.split()))
    starts, lengths = s[0::2], s[1::2]

    starts = np.asarray(starts) - 1           # CSV is 1-indexed
    ends   = starts + lengths

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1

    return mask.reshape(shape, order="F")     # column-major

# ---------- Overlay & save ------------------------------------------------- #


def save_preview(
    image: np.ndarray,
    mask: np.ndarray,
    out_path: str | Path,
    *,
    alpha: float = 0.35,
) -> None:
    """
    Blend `mask` (in red) onto `image` and save as PNG.

    Parameters
    ----------
    image : np.ndarray, shape (H, W) or (H, W, 3), uint8
        Grayscale or RGB source image.
    mask : np.ndarray, shape (H, W), {0,1}
        Binary segmentation mask to overlay.
    out_path : str | Path
        Where to write the PNG.
    alpha : float, default 0.35
        Opacity of the red overlay.
    """
    if image.ndim == 2:                       # gray → RGB
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image.copy()

    red = np.zeros_like(image_rgb)
    red[..., 0] = 255                         # pure-red layer

    blended = np.where(
        mask[..., None].astype(bool),
        (1 - alpha) * image_rgb + alpha * red,
        image_rgb,
    ).astype(np.uint8)

    Image.fromarray(blended).save(Path(out_path))
