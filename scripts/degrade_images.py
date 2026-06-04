"""
degrade_images.py - Image degradation pipeline for upscale LoRA training.

Creates paired training data:
  - Original sharp images -> TrainFolder (target/output)
  - Degraded images -> ControlFolder (condition/input)
"""

import argparse
import io
import sys
from pathlib import Path
from PIL import Image, ImageFilter


def degrade_bicubic(img: Image.Image, scale_factor: float) -> Image.Image:
    """Downscale then upscale via bicubic."""
    w, h = img.size
    small_w = max(1, int(w / scale_factor))
    small_h = max(1, int(h / scale_factor))
    small = img.resize((small_w, small_h), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def degrade_gaussian(img: Image.Image, radius: float = 2.0) -> Image.Image:
    """Apply Gaussian blur."""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def degrade_jpeg(img: Image.Image, quality: int = 30) -> Image.Image:
    """Compress to JPEG then decompress to add compression artifacts."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def degrade_combined(img: Image.Image, scale_factor: float,
                     blur_radius: float = 1.5, jpeg_quality: int = 40) -> Image.Image:
    """Apply all degradation steps in sequence."""
    result = degrade_bicubic(img, scale_factor)
    result = degrade_gaussian(result, blur_radius)
    result = degrade_jpeg(result, jpeg_quality)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Create degraded image pairs for upscale LoRA training"
    )
    parser.add_argument("--input", required=True, help="Input directory (sharp images)")
    parser.add_argument("--output", required=True, help="Output directory (degraded images)")
    parser.add_argument("--mode", default="combined",
                        choices=["bicubic_down", "gaussian_blur", "jpeg_artifact", "combined"],
                        help="Degradation mode")
    parser.add_argument("--scale-factor", type=float, default=2.0,
                        help="Downscale factor for bicubic_down/combined")
    parser.add_argument("--blur-radius", type=float, default=2.0,
                        help="Gaussian blur radius")
    parser.add_argument("--jpeg-quality", type=int, default=30,
                        help="JPEG quality for artifacts; lower means stronger artifacts")
    parser.add_argument("--format", default="jpg", choices=["jpg", "png", "webp"],
                        help="Output format")
    args = parser.parse_args()

    if args.scale_factor <= 1.0:
        parser.error("--scale-factor must be greater than 1.0")
    if args.blur_radius < 0:
        parser.error("--blur-radius cannot be negative")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")

    input_dir = Path(args.input)
    if not input_dir.exists() or not input_dir.is_dir():
        parser.error(f"Input path '{args.input}' does not exist or is not a directory.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]

    if not images:
        print(f"No valid images found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    stats = {"processed": 0, "errors": 0}

    for i, img_path in enumerate(images):
        try:
            img = Image.open(img_path).convert("RGB")

            if args.mode == "bicubic_down":
                degraded = degrade_bicubic(img, args.scale_factor)
            elif args.mode == "gaussian_blur":
                degraded = degrade_gaussian(img, args.blur_radius)
            elif args.mode == "jpeg_artifact":
                degraded = degrade_jpeg(img, args.jpeg_quality)
            else:
                degraded = degrade_combined(
                    img, args.scale_factor, args.blur_radius, args.jpeg_quality
                )

            out_path = output_dir / f"{img_path.stem}.{args.format}"

            if args.format == "jpg":
                degraded.save(out_path, "JPEG", quality=95)
            elif args.format == "webp":
                degraded.save(out_path, "WEBP", quality=95)
            else:
                degraded.save(out_path, "PNG")

            print(f"[{i + 1}/{len(images)}] OK {img_path.name} -> {out_path.name} ({args.mode})")
            stats["processed"] += 1

        except Exception as e:
            print(f"[{i + 1}/{len(images)}] ERROR {img_path.name}: {e}")
            stats["errors"] += 1

    print(f"\nStats: {stats['processed']} processed, {stats['errors']} errors")
    print(f"Use '{output_dir}' as ControlFolder in training config")


if __name__ == "__main__":
    main()
