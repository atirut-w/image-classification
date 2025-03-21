"""
Process the nether comparison dataset into train/test splits for image classification.

This script processes the nether comparison dataset (from either a directory or zip archive),
splits it into train/test sets, and organizes it into the required directory structure.

Two processing modes are supported:
1. Simple mode: 3 main categories (new_nether, old_nether, overworld)
2. Detailed mode: Individual biome types as separate classes

Directory structure for simple mode:
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
        ...

Directory structure for detailed mode:
dataset/
    train/
        basalt_deltas/
            image1.png
            ...
        crimson_forest/
            image1.png
            ...
        ...
    test/
        ...
"""

import os
import shutil
import random
import zipfile
import tempfile
from pathlib import Path
from sklearn.model_selection import train_test_split

def process_dataset(data_source='archive.zip', output_dir='dataset', test_size=0.2, random_state=42, detailed=False):
    """
    Process the nether comparison dataset into train/test splits
    
    Args:
        data_source: Path to the dataset (either a directory or a zip file)
        output_dir: Output directory for the processed dataset
        test_size: Fraction of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility
        detailed: If True, use detailed biome labels instead of general categories
    """
    print(f"Processing dataset from {data_source} to {output_dir}...")
    print(f"Mode: {'Detailed biomes' if detailed else 'Simple categories'}")
    random.seed(random_state)
    
    # Check if the data source is a zip file
    using_zip = data_source.lower().endswith('.zip')
    
    # If using a zip file, extract to a temporary directory
    if using_zip:
        if not os.path.exists(data_source):
            raise FileNotFoundError(f"Zip file not found: {data_source}")
        
        print(f"Extracting zip file: {data_source}")
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(data_source, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Based on the archive.zip structure, we know it has a 'data/nether_comparison_data' hierarchy
        data_path = os.path.join(temp_dir, 'data', 'nether_comparison_data')
        if os.path.exists(data_path):
            data_dir = data_path
        else:
            # Fallback - search for nether_comparison_data directory anywhere in the extracted contents
            found = False
            for root, dirs, _ in os.walk(temp_dir):
                if 'nether_comparison_data' in dirs:
                    data_dir = os.path.join(root, 'nether_comparison_data')
                    found = True
                    break
            
            if not found:
                # If still not found, use the temporary directory directly
                data_dir = temp_dir
                
        print(f"Using data directory: {data_dir}")
    else:
        # Using a directory directly
        data_dir = data_source
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if detailed:
        # In detailed mode, each biome folder becomes a separate class
        # First, discover available biomes
        biomes = []
        for root, dirs, _ in os.walk(data_dir):
            for dir_name in dirs:
                biome_path = os.path.join(root, dir_name)
                # Check if this is a biome directory (has PNG files)
                has_pngs = any(f.lower().endswith('.png') for f in os.listdir(biome_path))
                if has_pngs:
                    biomes.append((biome_path, dir_name))
        
        # Create necessary directories
        for subset in ['train', 'test']:
            for _, biome_name in biomes:
                os.makedirs(os.path.join(output_dir, subset, biome_name), exist_ok=True)
        
        # Collect all images by biome
        all_images = {}
        total_count = 0
        
        for biome_path, biome_name in biomes:
            all_images[biome_name] = []
            
            # Check if it's a direct biome directory (has PNG files directly)
            if any(f.lower().endswith('.png') for f in os.listdir(biome_path)):
                for file in os.listdir(biome_path):
                    if file.lower().endswith('.png'):
                        all_images[biome_name].append(os.path.join(biome_path, file))
            else:
                # It might be a parent directory containing biome subdirectories
                for root, _, files in os.walk(biome_path):
                    for file in files:
                        if file.lower().endswith('.png'):
                            all_images[biome_name].append(os.path.join(root, file))
            
            print(f"Found {len(all_images[biome_name])} images in {biome_name} biome")
            total_count += len(all_images[biome_name])
        
        print(f"Total images: {total_count}")
        
        # Split into train and test for each biome
        for biome_name, images in all_images.items():
            if len(images) < 2:
                print(f"Warning: Not enough images for {biome_name} to split into train/test")
                continue
                
            train_images, test_images = train_test_split(
                images,
                test_size=test_size,
                random_state=random_state
            )
            
            print(f"{biome_name}: {len(train_images)} training, {len(test_images)} testing")
            
            # Copy files to appropriate directories
            for subset, subset_images in [('train', train_images), ('test', test_images)]:
                for src in subset_images:
                    # Get just the filename without the directory structure
                    filename = os.path.basename(src)
                    dst = os.path.join(output_dir, subset, biome_name, filename)
                    shutil.copy(src, dst)
    
    else:
        # Original simple mode with 3 categories
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
    print("\nFinal dataset statistics:")
    for subset in ['train', 'test']:
        subset_dir = os.path.join(output_dir, subset)
        if not os.path.exists(subset_dir):
            continue
            
        for cls in sorted(os.listdir(subset_dir)):
            path = os.path.join(subset_dir, cls)
            if os.path.isdir(path):
                count = len(os.listdir(path))
                print(f"{subset}/{cls}: {count} images")
    
    print("Dataset processing complete!")

if __name__ == "__main__":
    # Check if this is running in a notebook environment
    try:
        # This will only be defined in IPython/Jupyter environments
        get_ipython()
        IN_NOTEBOOK = True
    except NameError:
        IN_NOTEBOOK = False
    
    # Use argparse only if not in a notebook
    if IN_NOTEBOOK:
        # Default values for notebook environment
        data_source = "archive.zip"
        output_dir = "dataset"
        test_size = 0.2
        random_seed = 42
        clean = True
        detailed = True
        
        # Clean output directory if requested
        if clean:
            output_path = os.path.join(output_dir)
            if os.path.exists(output_path):
                print(f"Cleaning {output_path}...")
                for subset in ['train', 'test']:
                    subset_path = os.path.join(output_path, subset)
                    if os.path.exists(subset_path):
                        for cls in os.listdir(subset_path):
                            class_path = os.path.join(subset_path, cls)
                            if os.path.isdir(class_path):
                                print(f"Removing {class_path}...")
                                shutil.rmtree(class_path)
        
        # Process the dataset
        process_dataset(
            data_source=data_source,
            output_dir=output_dir,
            test_size=test_size,
            random_state=random_seed,
            detailed=detailed
        )
    else:
        # Command-line usage
        import argparse
        
        parser = argparse.ArgumentParser(description="Process nether comparison dataset into train/test splits")
        parser.add_argument("--input", default="archive.zip", 
                          help="Source of the dataset (either a zip file or directory)")
        parser.add_argument("--output", default="dataset", help="Output directory")
        parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for testing")
        parser.add_argument("--seed", type=int, default=42, help="Random seed")
        parser.add_argument("--clean", action="store_true", help="Clean output directory before processing")
        parser.add_argument("--detailed", action="store_true", help="Use detailed biome labels instead of general categories")
        
        args = parser.parse_args()
        
        # Clean output directory if requested
        if args.clean:
            output_path = os.path.join(args.output)
            if os.path.exists(output_path):
                print(f"Cleaning {output_path}...")
                for subset in ['train', 'test']:
                    subset_path = os.path.join(output_path, subset)
                    if os.path.exists(subset_path):
                        for cls in os.listdir(subset_path):
                            class_path = os.path.join(subset_path, cls)
                            if os.path.isdir(class_path):
                                print(f"Removing {class_path}...")
                                shutil.rmtree(class_path)
        
        process_dataset(
            data_source=args.input,
            output_dir=args.output,
            test_size=args.test_size,
            random_state=args.seed,
            detailed=args.detailed
        )