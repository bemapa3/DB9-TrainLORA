"""
prepare_upscale_detail_dataset.py - Build tile pairs for upscale/detail LoRA training.

Creates:
  output/img       sharp target tiles
  output/control   degraded input tiles with matching filenames
  output/captions  caption txt files with matching stems
"""

import argparse
import io
import math
import sys
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def degrade_bicubic(img: Image.Image, scale_factor: float) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, int(w / scale_factor)), max(1, int(h / scale_factor))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def degrade_jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def degrade_tile(img: Image.Image, scale_factor: float, blur_radius: float, jpeg_quality: int) -> Image.Image:
    degraded = degrade_bicubic(img, scale_factor)
    if blur_radius > 0:
        degraded = degraded.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return degrade_jpeg(degraded, jpeg_quality)


def detail_score(img: Image.Image) -> float:
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    return float(ImageStat.Stat(edges).stddev[0])


def tile_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def save_tile(tile: Image.Image, path: Path, fmt: str, quality: int) -> None:
    if fmt == "jpg":
        tile.save(path, "JPEG", quality=quality)
    elif fmt == "webp":
        tile.save(path, "WEBP", quality=quality)
    else:
        tile.save(path, "PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare tiled upscale/detail training pairs")
    parser.add_argument("--input", required=True, help="Input folder with high-resolution source images")
    parser.add_argument("--output", default="datasets/processed", help="Output dataset folder")
    parser.add_argument("--tile-size", type=int, default=2048, help="Tile size in pixels")
    parser.add_argument("--overlap", type=int, default=256, help="Tile overlap in pixels")
    parser.add_argument("--min-size", type=int, default=1024, help="Skip images smaller than this on either side")
    parser.add_argument("--min-detail-score", type=float, default=3.0, help="Skip low-detail tiles below this score")
    parser.add_argument("--max-tiles-per-image", type=int, default=0, help="0 means no limit")
    parser.add_argument("--degrade-scale", type=float, default=2.0, help="Downscale factor before upscaling control tile")
    parser.add_argument("--blur-radius", type=float, default=1.2, help="Gaussian blur radius for control tile")
    parser.add_argument("--jpeg-quality", type=int, default=45, help="JPEG quality for degraded control tile")
    parser.add_argument("--caption", default="high detail upscale restoration, sharp architectural detail", help="Caption text for each tile")
    parser.add_argument("--format", default="jpg", choices=["jpg", "png", "webp"], help="Output image format")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output tiles")
    args = parser.parse_args()

    if args.tile_size <= 0 or args.tile_size > 2048:
        parser.error("--tile-size must be between 1 and 2048 for Flux 2 Klein")
    if args.overlap < 0 or args.overlap >= args.tile_size:
        parser.error("--overlap must be >= 0 and smaller than --tile-size")
    if args.degrade_scale <= 1.0:
        parser.error("--degrade-scale must be greater than 1.0")
    if args.blur_radius < 0:
        parser.error("--blur-radius cannot be negative")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")

    input_dir = Path(args.input)
    if not input_dir.exists() or not input_dir.is_dir():
        parser.error(f"Input folder does not exist: {input_dir}")

    output_dir = Path(args.output)
    img_dir = output_dir / "img"
    control_dir = output_dir / "control"
    cap_dir = output_dir / "captions"
    img_dir.mkdir(parents=True, exist_ok=True)
    control_dir.mkdir(parents=True, exist_ok=True)
    cap_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])
    if not images:
        print(f"No valid images found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    stride = args.tile_size - args.overlap
    stats = {"images": 0, "tiles": 0, "skipped_small": 0, "skipped_detail": 0, "errors": 0}

    for image_path in images:
        try:
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
            if w < args.min_size or h < args.min_size:
                print(f"SKIP {image_path.name}: too small {w}x{h}")
                stats["skipped_small"] += 1
                continue

            stats["images"] += 1
            xs = tile_positions(w, min(args.tile_size, w), stride)
            ys = tile_positions(h, min(args.tile_size, h), stride)
            made_for_image = 0

            for y in ys:
                for x in xs:
                    if args.max_tiles_per_image and made_for_image >= args.max_tiles_per_image:
                        break

                    box = (x, y, min(x + args.tile_size, w), min(y + args.tile_size, h))
                    tile = img.crop(box)
                    if tile.size != (args.tile_size, args.tile_size):
                        canvas = Image.new("RGB", (args.tile_size, args.tile_size), (0, 0, 0))
                        canvas.paste(tile, (0, 0))
                        tile = canvas

                    score = detail_score(tile)
                    if score < args.min_detail_score:
                        stats["skipped_detail"] += 1
                        continue

                    stem = f"{image_path.stem}_x{x}_y{y}_s{args.tile_size}"
                    target_path = img_dir / f"{stem}.{args.format}"
                    control_path = control_dir / f"{stem}.{args.format}"
                    caption_path = cap_dir / f"{stem}.txt"

                    if target_path.exists() and control_path.exists() and caption_path.exists() and not args.overwrite:
                        continue

                    control = degrade_tile(tile, args.degrade_scale, args.blur_radius, args.jpeg_quality)
                    save_tile(tile, target_path, args.format, quality=95)
                    save_tile(control, control_path, args.format, quality=95)
                    caption_path.write_text(args.caption, encoding="utf-8")
                    made_for_image += 1
                    stats["tiles"] += 1

                if args.max_tiles_per_image and made_for_image >= args.max_tiles_per_image:
                    break

            print(f"OK {image_path.name}: {made_for_image} tiles")
        except Exception as exc:
            print(f"ERROR {image_path.name}: {exc}")
            stats["errors"] += 1

    print(
        f"Done: {stats['images']} images, {stats['tiles']} tiles, "
        f"{stats['skipped_small']} small images skipped, "
        f"{stats['skipped_detail']} low-detail tiles skipped, {stats['errors']} errors"
    )
    print(f"Target tiles: {img_dir}")
    print(f"Control tiles: {control_dir}")
    print(f"Captions: {cap_dir}")


if __name__ == "__main__":
    main()
