"""
Flatten SIIM DICOM directory tree into data/images/
Choose one of: --copy  --hardlink  --symlink   (default = copy on Windows)

Example:
    python tools/flatten_dicom_tree.py --hardlink
    We used --hardlink to save disk space, but it requires that the source and destination are on the same drive.
"""
from pathlib import Path
import argparse, shutil, os, sys

ROOTS   = [Path("data/dicom-images-train"), Path("data/dicom-images-test")]
FLATDIR = Path("data/images")
FLATDIR.mkdir(parents=True, exist_ok=True)

def link(src: Path, dst: Path, mode: str):
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)           # works only if src & dst are on same drive
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"unknown mode {mode}")

def main(mode: str):
    for root in ROOTS:
        for dcm_path in root.rglob("*.dcm"):
            uid = dcm_path.stem
            dst = FLATDIR / f"{uid}.dcm"
            if dst.exists():
                continue
            try:
                link(dcm_path, dst, mode)
            except Exception as e:
                print(f"⚠️  {e} — falling back to copy")
                shutil.copy2(dcm_path, dst)

if __name__ == "__main__":
    default_mode = "copy" if sys.platform.startswith("win") else "symlink"
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--copy",     action="store_true")
    g.add_argument("--hardlink", action="store_true")
    g.add_argument("--symlink",  action="store_true")
    args = ap.parse_args()

    mode = ("hardlink" if args.hardlink else
            "symlink"  if args.symlink  else
            "copy"     if args.copy     else
            default_mode)
    print(f"Flattening with mode → {mode}")
    main(mode)
