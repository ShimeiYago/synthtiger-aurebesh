#!/usr/bin/env python3
"""
Export SynthTIGER output to PaddleOCR format.

This script converts SynthTIGER output to PaddleOCR format:
- Splits gt.txt into train.txt (90%) and val.txt (10%)
- Copies images directory
- Creates dict.txt with unique characters (excluding spaces)
"""

import argparse
import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple, Set


def read_gt_file(gt_path: str) -> List[Tuple[str, str]]:
    """Read ground truth file and return list of (image_path, text) tuples."""
    data = []
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split by tab: image_path \t text
            parts = line.split('\t')
            if len(parts) >= 2:
                image_path = parts[0]
                text = '\t'.join(parts[1:])  # Handle text with tabs
                data.append((image_path, text))
    
    return data


def split_data(data: List[Tuple[str, str]], train_ratio: float = 0.9) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Split data into train and validation sets."""
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    split_idx = int(len(shuffled_data) * train_ratio)
    train_data = shuffled_data[:split_idx]
    val_data = shuffled_data[split_idx:]
    
    return train_data, val_data


def write_gt_file(data: List[Tuple[str, str]], output_path: str):
    """Write ground truth data to file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for image_path, text in data:
            f.write(f"{image_path}\t{text}\n")


def extract_characters(data: List[Tuple[str, str]]) -> Set[str]:
    """Extract unique characters from all text labels, excluding spaces."""
    characters = set()
    for _, text in data:
        for char in text:
            if char != ' ':  # Exclude spaces
                characters.add(char)
    return characters


def write_dict_file(characters: Set[str], output_path: str):
    """Write character dictionary to file."""
    sorted_chars = sorted(list(characters))
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, char in enumerate(sorted_chars):
            if i < len(sorted_chars) - 1:
                f.write(f"{char}\n")
            else:
                f.write(char)  # Last character without newline


def copy_images(src_dir: str, dst_dir: str):
    """Copy images directory."""
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)


def main():
    parser = argparse.ArgumentParser(description='Export SynthTIGER output to PaddleOCR format')
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='results/synthtiger_output',
        help='Input directory containing SynthTIGER output (default: results/synthtiger_output)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results/paddleocr_data',
        help='Output directory for PaddleOCR format data (default: results/paddleocr_data)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.9,
        help='Ratio of training data (default: 0.9)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for data splitting (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Input paths
    input_dir = Path(args.input)
    gt_path = input_dir / 'gt.txt'
    images_dir = input_dir / 'images'
    
    # Output paths
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / 'train.txt'
    val_path = output_dir / 'val.txt'
    dict_path = output_dir / 'dict.txt'
    output_images_dir = output_dir / 'images'
    
    # Check input files exist
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    print(f"Reading ground truth from: {gt_path}")
    
    # Read ground truth data
    data = read_gt_file(str(gt_path))
    print(f"Total samples: {len(data)}")
    
    # Split data
    train_data, val_data = split_data(data, args.train_ratio)
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Write train and validation files
    print(f"Writing train data to: {train_path}")
    write_gt_file(train_data, str(train_path))
    
    print(f"Writing validation data to: {val_path}")
    write_gt_file(val_data, str(val_path))
    
    # Extract characters and create dictionary
    characters = extract_characters(data)
    print(f"Unique characters (excluding spaces): {len(characters)}")
    
    print(f"Writing character dictionary to: {dict_path}")
    write_dict_file(characters, str(dict_path))
    
    # Copy images
    print(f"Copying images from {images_dir} to {output_images_dir}")
    copy_images(str(images_dir), str(output_images_dir))
    
    print("\nExport completed successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print(f"  - {dict_path}")
    print(f"  - {output_images_dir}/")


if __name__ == '__main__':
    main()