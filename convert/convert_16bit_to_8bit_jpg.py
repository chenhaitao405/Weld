#!/usr/bin/env python
"""
Convert 16-bit grayscale images (TIFF/DICONDE) to 8-bit JPG.

- Recursively scans input directory for supported files.
- Output filenames are prefixed with relative subfolder path.
- Output images are saved into a single flat output directory.
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

DICOM_EXTS = {".dcm", ".dicom", ".dic", ".diconde"}
TIFF_EXTS = {".tif", ".tiff"}


def _safe_rel_prefix(path: Path, root: Path, sep: str) -> str:
    rel = path.parent.relative_to(root)
    if rel.parts:
        return sep.join(rel.parts) + sep
    return ""


def _normalize_to_uint8(img: np.ndarray,
                         clip_percent: Optional[Tuple[float, float]] = None,
                         invert: bool = False) -> np.ndarray:
    if img is None:
        raise ValueError("Empty image array")
    arr = img.astype(np.float32)

    if clip_percent is not None:
        low, high = np.percentile(arr, clip_percent)
    else:
        low, high = float(np.min(arr)), float(np.max(arr))

    if high <= low:
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = np.clip(arr, low, high)
    arr = (arr - low) / (high - low) * 255.0
    if invert:
        arr = 255.0 - arr
    return arr.astype(np.uint8)


def _read_tiff(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        # If accidentally loaded as RGB, convert to gray.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _read_dicom(path: Path) -> Optional[np.ndarray]:
    try:
        import pydicom
    except Exception as exc:
        raise RuntimeError("pydicom not available for DICOM/DICONDE reading") from exc

    ds = pydicom.dcmread(str(path), force=True)
    if not hasattr(ds, "PixelData"):
        return None

    arr = ds.pixel_array
    # Handle multi-frame (frames, rows, cols)
    if arr.ndim == 3:
        arr = arr[0]
    elif arr.ndim == 4:
        arr = arr[0, 0]

    arr = arr.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    invert = False
    photometric = getattr(ds, "PhotometricInterpretation", "").upper()
    if photometric == "MONOCHROME1":
        invert = True

    return arr, invert


def _iter_supported_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in TIFF_EXTS or ext in DICOM_EXTS:
            files.append(path)
    return files


def convert_folder(input_dir: Path,
                   output_dir: Path,
                   clip_percent: Optional[Tuple[float, float]],
                   overwrite: bool,
                   sep: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_supported_files(input_dir)
    if not files:
        print(f"No supported files found under {input_dir}")
        return

    print(f"Found {len(files)} files")

    skipped = 0
    failed = 0
    converted = 0

    iterable = tqdm(files, desc="Converting", unit="file") if tqdm else files
    for path in iterable:
        ext = path.suffix.lower()
        prefix = _safe_rel_prefix(path, input_dir, sep)
        out_name = f"{prefix}{path.stem}.jpg"
        out_path = output_dir / out_name

        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            invert = False
            if ext in TIFF_EXTS:
                img = _read_tiff(path)
                if img is None:
                    raise RuntimeError("Failed to read TIFF")
            else:
                result = _read_dicom(path)
                if result is None:
                    raise RuntimeError("No PixelData in DICOM")
                img, invert = result

            img_8 = _normalize_to_uint8(img, clip_percent=clip_percent, invert=invert)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(out_path), img_8, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise RuntimeError("cv2.imwrite failed")
            converted += 1
        except Exception as exc:
            failed += 1
            print(f"[WARN] {path}: {exc}")

    print(f"Converted: {converted}, Skipped: {skipped}, Failed: {failed}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert 16-bit TIFF/DICONDE images to 8-bit JPG with folder-prefix filenames."
    )
    parser.add_argument(
        "--input-dir",
        default="/datasets/PAR/Xray/self/nocrack",
        help="Root folder to scan (default: /datasets/PAR/Xray/self/nocrack)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder for JPGs (default: <input-dir>_jpg8)",
    )
    parser.add_argument(
        "--clip-percent",
        nargs=2,
        type=float,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Percentile clipping before scaling, e.g. --clip-percent 0.5 99.5",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JPGs",
    )
    parser.add_argument(
        "--sep",
        default="__",
        help="Separator between folder prefix parts (default: '__')",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(str(input_dir) + "_jpg8")

    clip_percent = None
    if args.clip_percent is not None:
        if len(args.clip_percent) != 2:
            raise ValueError("--clip-percent requires two values")
        clip_percent = (min(args.clip_percent), max(args.clip_percent))

    convert_folder(input_dir, output_dir, clip_percent, args.overwrite, args.sep)


if __name__ == "__main__":
    main()
