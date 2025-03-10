"""
Process the mc-fakes dataset into train/test splits for image classification.

This script reads the mc-fakes.csv file, splits the dataset into 
train/test sets, and organizes them into the required directory structure:

dataset/
    train/
        real/
            image1.jpg
            ...
        fake/
            image1.jpg
            ...
    test/
        real/
            image1.jpg
            ...
        fake/
            image1.jpg
            ...
"""

import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

def process_dataset(mc_fakes_dir='mc-fakes', output_dir='dataset', test_size=0.2, random_state=42):
    """
    Process the mc-fakes dataset into train/test splits
    
    Args:
        mc_fakes_dir: Directory containing the mc-fakes dataset and CSV
        output_dir: Output directory for the processed dataset
        test_size: Fraction of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility
    """
    print(f"Processing dataset from {mc_fakes_dir} to {output_dir}...")
    
    # Create necessary directories
    for subset in ['train', 'test']:
        for cls in ['real', 'fake']:
            os.makedirs(os.path.join(output_dir, subset, cls), exist_ok=True)
    
    # Read the CSV
    csv_path = os.path.join(mc_fakes_dir, 'mc-fakes.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} images in CSV file")
    
    # Map the labels
    df['label'] = df['fake'].map({0: 'real', 1: 'fake'})
    
    # Count by class
    class_counts = df['label'].value_counts()
    print(f"Class distribution: {class_counts.to_dict()}")
    
    # Split into train and test
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['label'], 
        random_state=random_state
    )
    
    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")
    
    # Copy files to appropriate directories
    for subset, subset_df in [('train', train_df), ('test', test_df)]:
        for idx, row in subset_df.iterrows():
            src = os.path.join(mc_fakes_dir, row['filename'])
            dst = os.path.join(output_dir, subset, row['label'], row['filename'])
            if not os.path.exists(src):
                print(f"Warning: Source file not found: {src}")
                continue
            shutil.copy(src, dst)
    
    # Count files in each directory
    for subset in ['train', 'test']:
        for cls in ['real', 'fake']:
            path = os.path.join(output_dir, subset, cls)
            count = len(os.listdir(path))
            print(f"{subset}/{cls}: {count} images")
    
    print("Dataset processing complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process mc-fakes dataset into train/test splits")
    parser.add_argument("--input", default="mc-fakes", help="Directory containing mc-fakes dataset")
    parser.add_argument("--output", default="dataset", help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--clean", action="store_true", help="Clean output directory before processing")
    
    args = parser.parse_args()
    
    # Clean output directory if requested
    if args.clean:
        for subset in ['train', 'test']:
            for cls in ['real', 'fake']:
                path = os.path.join(args.output, subset, cls)
                if os.path.exists(path):
                    print(f"Cleaning {path}...")
                    shutil.rmtree(path)
    
    process_dataset(
        mc_fakes_dir=args.input,
        output_dir=args.output,
        test_size=args.test_size,
        random_state=args.seed
    )