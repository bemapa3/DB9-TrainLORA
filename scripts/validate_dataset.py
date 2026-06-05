"""
validate_dataset.py - Validate DB9 processed dataset filenames before training.

Checks matching stems across img/captions/control and reports bad or missing names.
"""

import argparse
import re
import sys
from pathlib import Path

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SAFE_STEM_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def collect_images(path: Path) -> dict[str, Path]:
    if not path.exists():
        return {}
    return {p.stem: p for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() in VALID_EXTS}


def collect_captions(path: Path) -> dict[str, Path]:
    if not path.exists():
        return {}
    return {p.stem: p for p in sorted(path.glob("*.txt")) if p.is_file()}


def log_bad_stems(label: str, files: dict[str, Path]) -> int:
    bad = 0
    for stem, path in files.items():
        if not SAFE_STEM_RE.match(stem):
            print(f"BAD_NAME {label}: {path.name} | use only letters, numbers, dot, dash, underscore")
            bad += 1
    return bad


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate processed dataset filename matching before training")
    parser.add_argument("--processed", default="datasets/processed", help="Processed dataset folder")
    parser.add_argument("--mode", default="text2img", choices=["text2img", "img2img_upscale", "img2img_kontext"])
    parser.add_argument("--control", default="", help="Control folder for img2img/upscale modes")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on warnings")
    args = parser.parse_args()

    processed = Path(args.processed)
    img_dir = processed / "img"
    cap_dir = processed / "captions"
    control_dir = Path(args.control) if args.control else processed / "control"

    images = collect_images(img_dir)
    captions = collect_captions(cap_dir)
    controls = collect_images(control_dir)

    warnings = 0
    if not images:
        print(f"ERROR no target images found: {img_dir}")
        sys.exit(2)

    warnings += log_bad_stems("img", images)
    warnings += log_bad_stems("caption", captions)
    if args.mode != "text2img":
        warnings += log_bad_stems("control", controls)

    for stem, path in images.items():
        if stem not in captions:
            print(f"BAD_NAME missing caption: {path.name} expects {stem}.txt")
            warnings += 1
        if args.mode != "text2img" and stem not in controls:
            print(f"BAD_NAME missing control: {path.name} expects matching control image stem '{stem}'")
            warnings += 1

    for stem, path in captions.items():
        if stem not in images:
            print(f"BAD_NAME orphan caption: {path.name} has no matching target image")
            warnings += 1

    if args.mode != "text2img":
        for stem, path in controls.items():
            if stem not in images:
                print(f"BAD_NAME orphan control: {path.name} has no matching target image")
                warnings += 1

    print(f"Dataset validation: {len(images)} targets, {len(captions)} captions, {len(controls)} controls, {warnings} warnings")
    if warnings and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()