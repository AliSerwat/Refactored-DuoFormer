#!/usr/bin/env python3
"""
🔍 Refactored DuoFormer - Data Structure Validator

Validates data directory structure before training to prevent errors.
Run this script to check if your data is properly organized.

Usage:
    python scripts/validate_data.py --data_dir /path/to/data
    python scripts/validate_data.py --csv_file data.csv

Original work: https://github.com/xiaoyatang/duoformer_TCGA
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def validate_directory_structure(data_dir: str) -> Tuple[bool, List[str], Dict[str, int]]:
    """
    Validate directory structure for medical image classification.

    Args:
        data_dir: Path to data directory

    Returns:
        Tuple of (is_valid, issues, class_counts)
    """
    data_path = Path(data_dir)
    issues = []
    class_counts = {}

    print(f"🔍 Validating directory: {data_dir}")

    # Check if directory exists
    if not data_path.exists():
        issues.append(f"Directory does not exist: {data_dir}")
        return False, issues, class_counts

    if not data_path.is_dir():
        issues.append(f"Path is not a directory: {data_dir}")
        return False, issues, class_counts

    # Check for class subdirectories
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]

    if not class_dirs:
        issues.append("No class subdirectories found")
        return False, issues, class_counts

    print(f"📁 Found {len(class_dirs)} class directories:")

    # Validate each class directory
    supported_formats = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        print(f"   📂 {class_name}/")

        # Check for images in class directory
        image_files = []
        for file_path in class_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                image_files.append(file_path)

        if not image_files:
            issues.append(f"No images found in class directory: {class_name}")
            continue

        class_counts[class_name] = len(image_files)
        print(f"      ✅ {len(image_files)} images found")

        # Validate a few sample images
        sample_images = image_files[:3]  # Check first 3 images
        for img_path in sample_images:
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                issues.append(f"Corrupted image: {img_path} - {e}")

    return len(issues) == 0, issues, class_counts


def validate_csv_structure(csv_file: str) -> Tuple[bool, List[str], Dict[str, int]]:
    """
    Validate CSV file structure for medical image classification.

    Args:
        csv_file: Path to CSV file

    Returns:
        Tuple of (is_valid, issues, class_counts)
    """
    issues = []
    class_counts = {}

    print(f"🔍 Validating CSV file: {csv_file}")

    csv_path = Path(csv_file)

    # Check if file exists
    if not csv_path.exists():
        issues.append(f"CSV file does not exist: {csv_file}")
        return False, issues, class_counts

    if not csv_path.is_file():
        issues.append(f"Path is not a file: {csv_file}")
        return False, issues, class_counts

    try:
        # Load CSV
        df = pd.read_csv(csv_file)
        print(f"📄 CSV loaded: {len(df)} rows")

        # Check required columns
        required_columns = ['image_path', 'label']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
            return False, issues, class_counts

        print("✅ Required columns found: image_path, label")

        # Check for empty values
        empty_paths = df['image_path'].isna().sum()
        empty_labels = df['label'].isna().sum()

        if empty_paths > 0:
            issues.append(f"Empty image paths: {empty_paths} rows")

        if empty_labels > 0:
            issues.append(f"Empty labels: {empty_labels} rows")

        # Count classes
        class_counts = df['label'].value_counts().to_dict()
        print(f"📊 Found {len(class_counts)} classes:")

        for class_name, count in class_counts.items():
            print(f"   📂 {class_name}: {count} images")

        # Validate sample image paths
        sample_paths = df['image_path'].head(5)
        supported_formats = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

        for img_path in sample_paths:
            img_path_obj = Path(img_path)

            # Check if file exists
            if not img_path_obj.exists():
                issues.append(f"Image file not found: {img_path}")
                continue

            # Check file format
            if img_path_obj.suffix.lower() not in supported_formats:
                issues.append(f"Unsupported image format: {img_path}")
                continue

            # Try to open image
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                issues.append(f"Corrupted image: {img_path} - {e}")

    except Exception as e:
        issues.append(f"Error reading CSV file: {e}")
        return False, issues, class_counts

    return len(issues) == 0, issues, class_counts


def print_data_structure_guide():
    """Print the required data structure guide."""
    print("\n" + "=" * 80)
    print("🌳 REQUIRED DATA STRUCTURE GUIDE")
    print("=" * 80)
    print("\n📂 METHOD 1: Directory Structure (Recommended)")
    print("   ┌─────────────────────────────────────────────────────────┐")
    print("   │                    your_data_folder/                    │")
    print("   │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────┐ │")
    print("   │  │   class_name_1  │  │   class_name_2  │  │ class_3 │ │")
    print("   │  │  ┌─────────────┐ │  │  ┌─────────────┐ │ │ ┌─────┐ │ │")
    print("   │  │  │ image1.png  │ │  │  │ image1.jpg  │ │ │ │img1 │ │ │")
    print("   │  │  │ image2.jpg  │ │  │  │ image2.tif  │ │ │ │img2 │ │ │")
    print("   │  │  │ image3.tif  │ │  │  │ image3.png  │ │ │ │img3 │ │ │")
    print("   │  │  └─────────────┘ │  │  └─────────────┘ │ │ └─────┘ │ │")
    print("   │  └─────────────────┘  └─────────────────┘  └─────────┘ │")
    print("   └─────────────────────────────────────────────────────────┘")
    print("\n📋 STRUCTURE RULES:")
    print("   ✅ Each class = separate folder")
    print("   ✅ Class names = descriptive (benign, malignant, normal)")
    print("   ✅ Images = PNG, JPG, JPEG, TIF, TIFF formats")
    print("   ✅ No spaces in class folder names (use underscores)")
    print("\n💡 EXAMPLE FOR MEDICAL IMAGING:")
    print("   medical_data/")
    print("   ├── benign/")
    print("   │   ├── patient_001.png")
    print("   │   └── patient_002.jpg")
    print("   ├── malignant/")
    print("   │   ├── tumor_001.tif")
    print("   │   └── tumor_002.png")
    print("   └── normal/")
    print("       └── healthy_001.jpg")
    print("\n📄 METHOD 2: CSV File")
    print("   image_path,label")
    print("   /path/to/image1.png,benign")
    print("   /path/to/image2.jpg,malignant")
    print("   /path/to/image3.tif,normal")
    print("=" * 80)


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description="Validate data structure for DuoFormer training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_data.py --data_dir ./medical_data
  python scripts/validate_data.py --csv_file data.csv
  python scripts/validate_data.py --show-guide
        """
    )

    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    parser.add_argument("--csv_file", type=str, help="Path to CSV file")
    parser.add_argument("--show-guide", action="store_true", help="Show data structure guide")

    args = parser.parse_args()

    if args.show_guide:
        print_data_structure_guide()
        return

    if not args.data_dir and not args.csv_file:
        print("❌ Please provide either --data_dir or --csv_file")
        print("   Use --show-guide to see the required data structure")
        return

    print("🔍 Refactored DuoFormer - Data Structure Validator")
    print("=" * 60)

    is_valid = False
    issues = []
    class_counts = {}

    if args.data_dir:
        is_valid, issues, class_counts = validate_directory_structure(args.data_dir)
    elif args.csv_file:
        is_valid, issues, class_counts = validate_csv_structure(args.csv_file)

    print("\n" + "=" * 60)
    print("📊 VALIDATION RESULTS")
    print("=" * 60)

    if is_valid:
        print("✅ Data structure is VALID!")
        print(f"📊 Found {len(class_counts)} classes:")
        for class_name, count in class_counts.items():
            print(f"   • {class_name}: {count} images")
        print("\n🚀 Ready for training!")
        print("   python train.py --data_dir" + (f" {args.data_dir}" if args.data_dir else f" {args.csv_file}"))
    else:
        print("❌ Data structure has ISSUES:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n💡 Fix the issues above and run validation again")
        print("   Use --show-guide to see the required data structure")

    print("=" * 60)


if __name__ == "__main__":
    main()
