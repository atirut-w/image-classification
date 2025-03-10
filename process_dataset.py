"""
Process the nether comparison dataset into train/test splits for image classification.

This script processes the nether_comparison_data directory, splits the dataset into 
train/test sets, and organizes them into the required directory structure:

dataset/
    train/
        new_nether/
            image1.png
            ...
        old_nether/
            image1.png
            ...
        overworld/
            image1.png
            ...
    test/
        new_nether/
            image1.png
            ...
        old_nether/
            image1.png
            ...
        overworld/
            image1.png
            ...
"""

import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

def process_dataset(data_dir='nether_comparison_data', output_dir='dataset', test_size=0.2, random_state=42):
    """
    Process the nether comparison dataset into train/test splits
    
    Args:
        data_dir: Directory containing the nether comparison data
        output_dir: Output directory for the processed dataset
        test_size: Fraction of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility
    """
    print(f"Processing dataset from {data_dir} to {output_dir}...")
    random.seed(random_state)
    
    # Define class mappings
    class_mappings = {
        'new_nether_sc': 'new_nether',
        'old_nether_sc': 'old_nether',
        'overworld_sc': 'overworld'
    }
    
    # Create necessary directories
    for subset in ['train', 'test']:
        for cls in class_mappings.values():
            os.makedirs(os.path.join(output_dir, subset, cls), exist_ok=True)
    
    # Collect all images
    all_images = {}
    total_count = 0
    
    for src_dir, class_name in class_mappings.items():
        all_images[class_name] = []
        src_path = os.path.join(data_dir, src_dir)
        
        if not os.path.exists(src_path):
            print(f"Warning: Source directory not found: {src_path}")
            continue
        
        # Find all PNG files recursively
        for root, _, files in os.walk(src_path):
            for file in files:
                if file.lower().endswith('.png'):
                    all_images[class_name].append(os.path.join(root, file))
        
        print(f"Found {len(all_images[class_name])} images in {class_name} class")
        total_count += len(all_images[class_name])
    
    print(f"Total images: {total_count}")
    
    # Split into train and test for each class
    for class_name, images in all_images.items():
        train_images, test_images = train_test_split(
            images,
            test_size=test_size,
            random_state=random_state
        )
        
        print(f"{class_name}: {len(train_images)} training, {len(test_images)} testing")
        
        # Copy files to appropriate directories
        for subset, subset_images in [('train', train_images), ('test', test_images)]:
            for src in subset_images:
                # Get just the filename without the directory structure
                filename = os.path.basename(src)
                dst = os.path.join(output_dir, subset, class_name, filename)
                shutil.copy(src, dst)
    
    # Count files in each directory
    for subset in ['train', 'test']:
        for cls in class_mappings.values():
            path = os.path.join(output_dir, subset, cls)
            count = len(os.listdir(path))
            print(f"{subset}/{cls}: {count} images")
    
    print("Dataset processing complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process nether comparison dataset into train/test splits")
    parser.add_argument("--input", default="nether_comparison_data", help="Directory containing nether comparison data")
    parser.add_argument("--output", default="dataset", help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--clean", action="store_true", help="Clean output directory before processing")
    
    args = parser.parse_args()
    
    # Clean output directory if requested
    if args.clean:
        class_mappings = {
            'new_nether_sc': 'new_nether',
            'old_nether_sc': 'old_nether',
            'overworld_sc': 'overworld'
        }
        
        for subset in ['train', 'test']:
            for cls in class_mappings.values():
                path = os.path.join(args.output, subset, cls)
                if os.path.exists(path):
                    print(f"Cleaning {path}...")
                    shutil.rmtree(path)
    
    process_dataset(
        data_dir=args.input,
        output_dir=args.output,
        test_size=args.test_size,
        random_state=args.seed
    )