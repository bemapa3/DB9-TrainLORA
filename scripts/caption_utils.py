"""
caption_utils.py - Caption post-processing utilities.

Features:
  --add-trigger     : Prepend/append trigger word to all captions
  --add-folder-name : Add caption folder name as trigger word
  --remove          : Remove specific text from all captions
  --append          : Place additions at end instead of beginning

Usage:
  python scripts/caption_utils.py --input datasets/processed/captions --add-trigger "db9_style"
  python scripts/caption_utils.py --input datasets/processed/captions --add-folder-name
  python scripts/caption_utils.py --input datasets/processed/captions --remove "blurry"
"""

import argparse
import sys
from pathlib import Path


def process_captions(cap_dir: Path, add_trigger: str = "",
                     add_folder_name: bool = False,
                     remove_text: str = "",
                     append: bool = False):
    """Process all .txt caption files in directory."""
    txt_files = sorted(cap_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt caption files found in {cap_dir}", file=sys.stderr)
        return 0

    folder_name = cap_dir.name if add_folder_name else ""
    modified = 0

    for txt_path in txt_files:
        original = txt_path.read_text(encoding="utf-8").strip()
        result = original

        if remove_text:
            result = result.replace(remove_text, "").strip()
            while "  " in result:
                result = result.replace("  ", " ")
            while ",," in result:
                result = result.replace(",,", ",")
            result = result.strip(", ")

        if folder_name:
            if append:
                result = f"{result}, {folder_name}" if result else folder_name
            else:
                result = f"{folder_name}, {result}" if result else folder_name

        if add_trigger:
            if append:
                result = f"{result}, {add_trigger}" if result else add_trigger
            else:
                result = f"{add_trigger}, {result}" if result else add_trigger

        if result != original:
            txt_path.write_text(result, encoding="utf-8")
            modified += 1
            preview = result[:100].encode("ascii", "replace").decode("ascii")
            print(f"OK {txt_path.name}: {preview}...")

    return modified


def main():
    parser = argparse.ArgumentParser(
        description="Post-process caption .txt files"
    )
    parser.add_argument("--input", required=True,
                        help="Directory containing .txt caption files")
    parser.add_argument("--add-trigger", default="",
                        help="Trigger word to add to all captions")
    parser.add_argument("--add-folder-name", action="store_true",
                        help="Add parent folder name as trigger word")
    parser.add_argument("--remove", default="",
                        help="Text to remove from all captions")
    parser.add_argument("--append", action="store_true",
                        help="Add text at end instead of beginning")
    args = parser.parse_args()

    cap_dir = Path(args.input)
    if not cap_dir.exists() or not cap_dir.is_dir():
        parser.error(f"Input path '{args.input}' does not exist or is not a directory.")

    if not args.add_trigger and not args.add_folder_name and not args.remove:
        parser.error("At least one of --add-trigger, --add-folder-name, or --remove is required.")

    modified = process_captions(
        cap_dir,
        add_trigger=args.add_trigger,
        add_folder_name=args.add_folder_name,
        remove_text=args.remove,
        append=args.append,
    )

    print(f"\nModified {modified} caption files")


if __name__ == "__main__":
    main()
