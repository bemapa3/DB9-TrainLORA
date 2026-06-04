"""
caption_gemini.py - Gemini API Caption Generator (Multi-Model)

API keys are loaded only from .env / GEMINI_API_KEY. Do not pass secrets on the command line.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

CAPTION_PROMPTS = {
    "Short": (
        "Describe this image in 1-2 short sentences. "
        "Focus on the main subject and action."
    ),
    "Medium": (
        "Provide a medium-length description of this image. "
        "Include the main subject, setting, colors, lighting, and mood. "
        "Write 2-4 sentences."
    ),
    "Long": (
        "Provide a very detailed description of this image. "
        "Include subject details, clothing, pose, expression, background, "
        "colors, lighting, composition, art style, and mood. "
        "Write 4-8 detailed sentences."
    ),
}

SAFETY_FALLBACK_CAPTION = "image caption unavailable due to safety filtering"


def is_retryable_error(exc: Exception) -> bool:
    text = str(exc).lower()
    retry_markers = (
        "429",
        "rate limit",
        "quota",
        "resource exhausted",
        "timeout",
        "temporarily unavailable",
        "503",
        "500",
    )
    return any(marker in text for marker in retry_markers)


def generate_with_retry(model, payload, retries: int, base_delay: float):
    attempt = 0
    while True:
        try:
            return model.generate_content(payload)
        except Exception as exc:
            if attempt >= retries or not is_retryable_error(exc):
                raise
            delay = base_delay * (2 ** attempt)
            print(f"[Gemini] Retry {attempt + 1}/{retries} after {delay:.1f}s: {exc}")
            time.sleep(delay)
            attempt += 1


def extract_caption_or_fallback(response) -> tuple[str, bool]:
    if not getattr(response, "candidates", None):
        return SAFETY_FALLBACK_CAPTION, True

    try:
        text = response.text.strip()
    except Exception:
        return SAFETY_FALLBACK_CAPTION, True

    if not text:
        return SAFETY_FALLBACK_CAPTION, True

    return text, False


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Generate image captions using Gemini API (multi-model)"
    )
    parser.add_argument("--input", required=True, help="Input image directory")
    parser.add_argument("--output", help="Output caption directory (default: sibling captions folder)")
    parser.add_argument("--prompt", default="", help="Custom caption prompt (overrides --length)")
    parser.add_argument("--length", default="Medium", choices=["Short", "Medium", "Long"],
                        help="Caption length preset")
    parser.add_argument("--model", default="gemini-2.5-flash",
                        choices=["gemini-2.5-flash", "gemini-2.5-pro",
                                 "gemini-2.5-flash-lite", "gemini-1.5-flash"],
                        help="Gemini model to use")
    parser.add_argument("--rate-limit", type=float, default=1.0,
                        help="Seconds between successful requests")
    parser.add_argument("--retries", type=int, default=3,
                        help="Retry count for transient API/rate-limit errors")
    parser.add_argument("--retry-base-delay", type=float, default=2.0,
                        help="Base seconds for exponential backoff")
    parser.add_argument("--custom-caption", default="",
                        help="Trigger word to prepend to all captions")
    parser.add_argument("--append", action="store_true",
                        help="Append trigger word at end instead of beginning")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing captions")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        parser.error("GEMINI_API_KEY is required in .env or environment. Command-line API keys are disabled.")

    if args.retries < 0:
        parser.error("--retries cannot be negative")
    if args.retry_base_delay < 0:
        parser.error("--retry-base-delay cannot be negative")

    img_dir = Path(args.input)
    if not img_dir.exists() or not img_dir.is_dir():
        parser.error(f"Input path '{args.input}' does not exist or is not a directory.")

    cap_dir = Path(args.output) if args.output else img_dir.parent / "captions"
    if not args.output:
        print(f"[Gemini] No --output specified, using: {cap_dir}")
    cap_dir.mkdir(parents=True, exist_ok=True)

    caption_prompt = args.prompt if args.prompt else CAPTION_PROMPTS[args.length]
    print(f"[Gemini] Model: {args.model} | Length: {args.length}")
    print(f"[Gemini] Prompt: {caption_prompt[:80]}...")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted([p for p in img_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in valid_exts])

    if not images:
        print(f"No valid images found in {img_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[Gemini] Processing {len(images)} images...")
    stats = {"done": 0, "skip": 0, "fallback": 0, "error": 0}

    for i, img_path in enumerate(images):
        cap_path = cap_dir / f"{img_path.stem}.txt"

        if cap_path.exists() and not args.overwrite:
            print(f"[{i + 1}/{len(images)}] SKIP {img_path.name} (exists)")
            stats["skip"] += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((1024, 1024))

            response = generate_with_retry(
                model,
                [caption_prompt, img],
                retries=args.retries,
                base_delay=args.retry_base_delay,
            )
            text, used_fallback = extract_caption_or_fallback(response)

            if args.custom_caption:
                if args.append:
                    text = f"{text}, {args.custom_caption}"
                else:
                    text = f"{args.custom_caption}, {text}"

            cap_path.write_text(text, encoding="utf-8")
            preview = text[:60].encode("ascii", "replace").decode("ascii")
            status = "FALLBACK" if used_fallback else "OK"
            print(f"[{i + 1}/{len(images)}] {status} {img_path.name}: {preview}...")
            stats["fallback" if used_fallback else "done"] += 1

            time.sleep(args.rate_limit)
        except Exception as e:
            print(f"[{i + 1}/{len(images)}] ERROR {img_path.name}: {e}")
            stats["error"] += 1

    print(
        f"\nDone: {stats['done']} | Skipped: {stats['skip']} | "
        f"Fallback: {stats['fallback']} | Errors: {stats['error']}"
    )
    print(f"Total captions in {cap_dir}: {len(list(cap_dir.glob('*.txt')))}")


if __name__ == "__main__":
    main()
