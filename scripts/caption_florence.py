"""
caption_florence.py — Local Florence-2 Caption Generator

Uses Microsoft Florence-2-large for offline image captioning.
No API key needed — runs entirely on your GPU.

Caption lengths (matching DB9Studio workflow):
  Short  : <CAPTION>              (~10-30 tokens)
  Medium : <DETAILED_CAPTION>     (~10-100 tokens)
  Long   : <MORE_DETAILED_CAPTION> (~10-150 tokens)

Usage:
  python scripts/caption_florence.py --input datasets/processed/img --length Medium
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image


# Florence-2 task prompts by caption length
CAPTION_TASKS = {
    'Short': '<CAPTION>',
    'Medium': '<DETAILED_CAPTION>',
    'Long': '<MORE_DETAILED_CAPTION>',
}

# Cache the model globally so it's only loaded once
_FLORENCE_CACHE = {"model": None, "processor": None, "device": None}


def load_florence(model_name: str = "microsoft/Florence-2-large", device: str = "auto"):
    """Load Florence-2 model and processor. Cached after first call."""
    global _FLORENCE_CACHE
    
    if _FLORENCE_CACHE["model"] is not None:
        return _FLORENCE_CACHE["model"], _FLORENCE_CACHE["processor"], _FLORENCE_CACHE["device"]
    
    from transformers import AutoModelForCausalLM, AutoProcessor
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[Florence] Loading {model_name} on {device}...")
    
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device).eval()
    
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    _FLORENCE_CACHE.update(model=model, processor=processor, device=device)
    print(f"[Florence] Model loaded successfully.")
    return model, processor, device


def caption_image(model, processor, device, image: Image.Image, task_prompt: str) -> str:
    """Generate caption for a single image."""
    dtype = next(model.parameters()).dtype
    
    inputs = processor(
        text=task_prompt,
        images=image,
        return_tensors="pt",
    )
    # Move all tensors to device, but only cast pixel_values to model dtype
    # input_ids MUST stay as int64 — casting to float16 would crash
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # Parse Florence-2 output
    parsed = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height),
    )
    
    # Extract text from parsed result
    result = parsed.get(task_prompt, "")
    if isinstance(result, dict):
        result = result.get("text", str(result))
    
    return str(result).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Generate image captions using Florence-2 (local, no API)"
    )
    parser.add_argument('--input', required=True, help='Input image directory')
    parser.add_argument('--output', help='Output caption directory (default: same as input)')
    parser.add_argument('--length', default='Medium', choices=['Short', 'Medium', 'Long'],
                        help='Caption length (Short/Medium/Long)')
    parser.add_argument('--model', default='microsoft/Florence-2-large',
                        help='Florence model name or path')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Device to run on')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing captions')
    args = parser.parse_args()

    img_dir = Path(args.input)
    if not img_dir.exists() or not img_dir.is_dir():
        parser.error(f"Input path '{args.input}' does not exist or is not a directory.")

    if args.output:
        cap_dir = Path(args.output)
    else:
        # Default: sibling 'captions' folder (matches config_generator expectation)
        cap_dir = img_dir.parent / "captions"
        print(f"[Florence] No --output specified, using: {cap_dir}")
    cap_dir.mkdir(parents=True, exist_ok=True)

    task_prompt = CAPTION_TASKS[args.length]
    print(f"[Florence] Caption length: {args.length} -> Task: {task_prompt}")

    # Load model
    model, processor, device = load_florence(args.model, args.device)

    # Find images
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp'}
    images = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts])

    if not images:
        print(f"No valid images found in {img_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[Florence] Processing {len(images)} images...")

    for i, img_path in enumerate(images):
        cap_path = cap_dir / f"{img_path.stem}.txt"

        if cap_path.exists() and not args.overwrite:
            print(f"[{i+1}/{len(images)}] ⏭️  {img_path.name} (exists)")
            continue

        try:
            img = Image.open(img_path).convert('RGB')
            # Resize if too large to save VRAM
            img.thumbnail((1024, 1024))

            text = caption_image(model, processor, device, img, task_prompt)

            if not text:
                raise ValueError("Empty caption generated")

            cap_path.write_text(text, encoding='utf-8')
            print(f"[{i+1}/{len(images)}] ✅ {img_path.name}: {text[:80]}...")

        except Exception as e:
            print(f"[{i+1}/{len(images)}] ❌ {img_path.name}: {e}")

    print(f"\n✅ Completed: {len(list(cap_dir.glob('*.txt')))} captions in {cap_dir}")


if __name__ == '__main__':
    main()
