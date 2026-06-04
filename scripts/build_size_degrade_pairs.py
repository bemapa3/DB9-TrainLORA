"""
build_size_degrade_pairs.py - Build img2img upscale pairs from one sharp output folder and one degraded input folder.

Expected:
  output/sharp_name.jpg
  output/sharp_name.txt
  input/sharp_name_200.jpg
  input/sharp_name_2048.jpg

Creates:
  processed/img/sharp_name_200.jpg       copied from output/sharp_name.jpg
  processed/control/sharp_name_200.jpg   copied from input/sharp_name_200.jpg
  processed/captions/sharp_name_200.txt  copied from output/sharp_name.txt
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SIZE_SUFFIX_RE = re.compile(r"^(?P<base>.+)_(?P<size>\d+)$")


def parse_degrade_name(path: Path) -> tuple[str, str]:
    match = SIZE_SUFFIX_RE.match(path.stem)
    if not match:
        raise ValueError(f"Input file must end with _<size>, got: {path.name}")
    return match.group("base"), match.group("size")


def find_sharp(output_dir: Path, base: str) -> Path | None:
    for ext in VALID_EXTS:
        candidate = output_dir / f"{base}{ext}"
        if candidate.exists():
            return candidate
    lower_base = base.lower()
    for path in output_dir.iterdir():
        if path.is_file() and path.suffix.lower() in VALID_EXTS and path.stem.lower() == lower_base:
            return path
    return None


def copy_caption(output_dir: Path, base: str, target: Path, fallback_caption: str) -> None:
    source = output_dir / f"{base}.txt"
    if source.exists():
        shutil.copy2(source, target)
    else:
        target.write_text(fallback_caption, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-size degraded pairs for img2img upscale training")
    parser.add_argument("--input", required=True, help="Folder with degraded files named base_200.jpg, base_2048.jpg, ...")
    parser.add_argument("--output", required=True, help="Folder with sharp output images and optional base.txt captions")
    parser.add_argument("--processed", default="datasets/processed", help="Output processed dataset folder")
    parser.add_argument("--caption", default="upscale restoration, sharp high quality result", help="Fallback caption if output/base.txt is missing")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed files")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    processed_dir = Path(args.processed)
    if not input_dir.exists() or not input_dir.is_dir():
        parser.error(f"Input folder does not exist: {input_dir}")
    if not output_dir.exists() or not output_dir.is_dir():
        parser.error(f"Output folder does not exist: {output_dir}")

    img_dir = processed_dir / "img"
    control_dir = processed_dir / "control"
    cap_dir = processed_dir / "captions"
    img_dir.mkdir(parents=True, exist_ok=True)
    control_dir.mkdir(parents=True, exist_ok=True)
    cap_dir.mkdir(parents=True, exist_ok=True)

    degraded_files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])
    if not degraded_files:
        print(f"No degraded input files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    stats = {"pairs": 0, "missing_sharp": 0, "bad_name": 0, "skipped": 0}

    for degraded in degraded_files:
        try:
            base, size = parse_degrade_name(degraded)
        except ValueError as exc:
            print(f"SKIP {degraded.name}: {exc}")
            stats["bad_name"] += 1
            continue

        sharp = find_sharp(output_dir, base)
        if not sharp:
            print(f"SKIP {degraded.name}: missing sharp output for base '{base}'")
            stats["missing_sharp"] += 1
            continue

        pair_stem = f"{base}_{size}"
        target_img = img_dir / f"{pair_stem}{sharp.suffix.lower()}"
        target_control = control_dir / f"{pair_stem}{degraded.suffix.lower()}"
        target_caption = cap_dir / f"{pair_stem}.txt"

        if not args.overwrite and target_img.exists() and target_control.exists() and target_caption.exists():
            stats["skipped"] += 1
            continue

        shutil.copy2(sharp, target_img)
        shutil.copy2(degraded, target_control)
        copy_caption(output_dir, base, target_caption, args.caption)
        stats["pairs"] += 1
        print(f"OK {degraded.name} -> {pair_stem}")

    print(
        f"Done: {stats['pairs']} pairs, {stats['skipped']} skipped, "
        f"{stats['missing_sharp']} missing sharp, {stats['bad_name']} bad names"
    )
    print(f"Target folder: {img_dir}")
    print(f"Control folder: {control_dir}")
    print(f"Caption folder: {cap_dir}")


if __name__ == "__main__":
    main()
