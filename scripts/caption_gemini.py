"""
caption_gemini.py — Gemini API Caption Generator (Multi-Model)

Supported models (matching SDVN Colab):
  - gemini-2.5-flash      (fast, free tier)
  - gemini-2.5-pro        (high quality)
  - gemini-2.5-flash-lite  (ultra fast)
  - gemini-1.5-flash      (legacy)

Caption lengths:
  Short  : Brief 1-2 sentence description
  Medium : Detailed paragraph description  
  Long   : Very detailed multi-paragraph description

Usage:
  python scripts/caption_gemini.py --input datasets/img --output datasets/captions \\
    --model gemini-2.5-flash --length Medium --custom-caption "sdvn_style"
"""

import argparse
import os
import sys
import time
from pathlib import Path

import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv


# Caption length prompt templates
CAPTION_PROMPTS = {
    'Short': (
        "Describe this image in 1-2 short sentences. "
        "Focus on the main subject and action."
    ),
    'Medium': (
        "Provide a medium-length description of this image. "
        "Include the main subject, setting, colors, lighting, and mood. "
        "Write 2-4 sentences."
    ),
    'Long': (
        "Provide a very detailed description of this image. "
        "Include subject details, clothing, pose, expression, background, "
        "colors, lighting, composition, art style, and mood. "
        "Write 4-8 detailed sentences."
    ),
}


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Generate image captions using Gemini API (multi-model)"
    )
    parser.add_argument('--input', required=True, help='Input image directory')
    parser.add_argument('--output', help='Output caption directory (default: same as input)')
    parser.add_argument('--prompt', default='', help='Custom caption prompt (overrides --length)')
    parser.add_argument('--length', default='Medium', choices=['Short', 'Medium', 'Long'],
                        help='Caption length preset')
    parser.add_argument('--api-key', help='Gemini API key (or use GEMINI_API_KEY env)')
    parser.add_argument('--model', default='gemini-2.5-flash',
                        choices=['gemini-2.5-flash', 'gemini-2.5-pro',
                                 'gemini-2.5-flash-lite', 'gemini-1.5-flash'],
                        help='Gemini model to use')
    parser.add_argument('--rate-limit', type=float, default=1.0,
                        help='Seconds between requests')
    parser.add_argument('--custom-caption', default='',
                        help='Trigger word to prepend to all captions')
    parser.add_argument('--append', action='store_true',
                        help='Append trigger word at end instead of beginning')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing captions')
    args = parser.parse_args()

    # Setup API
    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
    if not api_key:
        parser.error("API key required. Provide --api-key or set GEMINI_API_KEY env var.")

    img_dir = Path(args.input)
    if not img_dir.exists() or not img_dir.is_dir():
        parser.error(f"Input path '{args.input}' does not exist or is not a directory.")

    if args.output:
        cap_dir = Path(args.output)
    else:
        # Default: sibling 'captions' folder (matches config_generator expectation)
        cap_dir = img_dir.parent / "captions"
        print(f"[Gemini] No --output specified, using: {cap_dir}")
    cap_dir.mkdir(parents=True, exist_ok=True)

    # Select prompt
    caption_prompt = args.prompt if args.prompt else CAPTION_PROMPTS[args.length]
    print(f"[Gemini] Model: {args.model} | Length: {args.length}")
    print(f"[Gemini] Prompt: {caption_prompt[:80]}...")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    # Find images
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp'}
    images = sorted([p for p in img_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in valid_exts])

    if not images:
        print(f"No valid images found in {img_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[Gemini] Processing {len(images)} images...")
    stats = {'done': 0, 'skip': 0, 'error': 0}

    for i, img_path in enumerate(images):
        cap_path = cap_dir / f"{img_path.stem}.txt"

        if cap_path.exists() and not args.overwrite:
            print(f"[{i+1}/{len(images)}] ⏭️  {img_path.name} (exists)")
            stats['skip'] += 1
            continue

        try:
            img = Image.open(img_path).convert('RGB')
            img.thumbnail((1024, 1024))

            response = model.generate_content([caption_prompt, img])

            # Check for safety blocks before accessing .text
            if not response.candidates:
                feedback = getattr(response, 'prompt_feedback', None)
                raise ValueError(f"Blocked by safety filter: {feedback}")

            text = response.text.strip()

            if not text:
                raise ValueError("Empty response from API")

            # Add custom caption / trigger word
            if args.custom_caption:
                if args.append:
                    text = f"{text}, {args.custom_caption}"
                else:
                    text = f"{args.custom_caption}, {text}"

            cap_path.write_text(text, encoding='utf-8')
            print(f"[{i+1}/{len(images)}] ✅ {img_path.name}: {text[:60]}...")
            stats['done'] += 1

            time.sleep(args.rate_limit)
        except Exception as e:
            print(f"[{i+1}/{len(images)}] ❌ {img_path.name}: {e}")
            stats['error'] += 1

    print(f"\n📊 Done: {stats['done']} | Skipped: {stats['skip']} | Errors: {stats['error']}")
    print(f"✅ Total captions in {cap_dir}: {len(list(cap_dir.glob('*.txt')))}")


if __name__ == '__main__':
    main()
