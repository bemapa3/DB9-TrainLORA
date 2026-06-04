import argparse
import sys
from pathlib import Path
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input image directory')
    parser.add_argument('--output', required=True, help='Output processed image directory')
    parser.add_argument('--min-res', type=int, default=512, help='Minimum resolution')
    parser.add_argument('--max-res', type=int, default=4096, help='Maximum resolution')
    parser.add_argument('--format', default='jpg', choices=['jpg', 'png', 'webp'], help='Output format')
    args = parser.parse_args()
    
    if args.min_res > args.max_res:
        parser.error("--min-res cannot be greater than --max-res")
        
    input_dir = Path(args.input)
    if not input_dir.exists() or not input_dir.is_dir():
        parser.error(f"Input path '{args.input}' does not exist or is not a directory.")
        
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp'}
    images = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    
    if not images:
        print(f"No valid images found in {input_dir}", file=sys.stderr)
        sys.exit(1)
        
    stats = {'processed': 0, 'skipped': 0, 'errors': 0}
    
    for img_path in images:
        try:
            # First verify the image is not corrupt
            with Image.open(img_path) as img:
                img.verify()
                
            # Now load it for processing
            img = Image.open(img_path)
            img.load()
            
            w, h = img.size
            
            # Check resolution
            if w < args.min_res or h < args.min_res:
                print(f"⏭️  {img_path.name}: too small ({w}x{h})")
                stats['skipped'] += 1
                continue
            
            if w > args.max_res or h > args.max_res:
                print(f"⏭️  {img_path.name}: too large ({w}x{h})")
                stats['skipped'] += 1
                continue
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                else:
                    img = img.convert('RGB')
            
            # Save
            output_path = output_dir / f"{img_path.stem}.{args.format}"
            
            if args.format == 'jpg':
                img.save(output_path, 'JPEG', quality=95)
            elif args.format == 'webp':
                img.save(output_path, 'WEBP', quality=95)
            else:
                img.save(output_path, 'PNG')
                
            print(f"✅ {img_path.name} -> {output_path.name}")
            stats['processed'] += 1
            
        except Exception as e:
            print(f"❌ {img_path.name}: {e}")
            stats['errors'] += 1
    
    print(f"\n📊 Stats: {stats['processed']} processed, {stats['skipped']} skipped, {stats['errors']} errors")

if __name__ == '__main__':
    main()
