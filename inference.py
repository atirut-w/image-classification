"""Run inference on images using trained image classification models

This script loads a trained model from a .h5 file and performs inference on 
individual images or a directory of images.

The script supports both:
1. Simple mode with 3 main categories (new_nether, old_nether, overworld)
2. Detailed mode with individual biome types

Usage:
    python inference.py --model MODEL_FILE --image IMAGE_PATH
    python inference.py --model MODEL_FILE --dir DIRECTORY_PATH

Arguments:
    --model: Path to the .h5 model file
    --image: Path to an image file to classify
    --dir: Path to a directory of images to classify
    --height: Image height (default: 128)
    --width: Image width (default: 128)
    --categories: Display high-level categories for biomes (useful with detailed biome mode)
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import json
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image

# Define category mappings for biomes
BIOME_CATEGORIES = {
    # Nether biomes
    "basalt_deltas": "new_nether",
    "crimson_forest": "new_nether",
    "nether_wastes": "new_nether",
    "soul_sand_valley": "new_nether",
    "warped_forest": "new_nether",
    
    # Overworld biomes
    "badlands": "overworld",
    "cave_ravine": "overworld",
    "dark_forest": "overworld",
    "desert": "overworld",
    "forest": "overworld",
    "giant_tree_taiga": "overworld",
    "jungle": "overworld",
    "mountains": "overworld",
    "mushroom_fields": "overworld",
    "ocean": "overworld",
    "plains": "overworld",
    "savanna": "overworld",
    "snowy_tundra": "overworld",
    "swamp": "overworld",
    "taiga": "overworld",
    
    # Already categorized
    "new_nether": "new_nether",
    "old_nether": "old_nether",
    "overworld": "overworld"
}

def load_model(model_path):
    """Load a trained model from a .h5 file"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return keras.models.load_model(model_path)

def preprocess_image(image_path, target_height=128, target_width=128):
    """Preprocess an image for inference"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load and resize the image
    img = Image.open(image_path)
    img = img.resize((target_width, target_height))
    
    # Convert to numpy array and normalize
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image(model, image_path, class_names, target_height=128, target_width=128, show_categories=False):
    """Run inference on a single image"""
    try:
        # Preprocess the image
        img_array = preprocess_image(image_path, target_height, target_width)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        # Get the category if showing categories
        category = BIOME_CATEGORIES.get(predicted_class, "unknown") if show_categories else None
        
        # Display results
        print(f"Image: {os.path.basename(image_path)}")
        if show_categories:
            print(f"Predicted: {predicted_class} ({category}) (Confidence: {confidence:.2%})")
        else:
            print(f"Predicted: {predicted_class} (Confidence: {confidence:.2%})")
        print("Top predictions:")
        
        # Get top 5 predictions or all if fewer than 5 classes
        top_n = min(5, len(class_names))
        top_indices = np.argsort(predictions[0])[-top_n:][::-1]
        for i in top_indices:
            if show_categories:
                category = BIOME_CATEGORIES.get(class_names[i], "unknown")
                print(f"  - {class_names[i]} ({category}): {predictions[0][i]:.2%}")
            else:
                print(f"  - {class_names[i]}: {predictions[0][i]:.2%}")
        
        # Group predictions by category if showing categories
        if show_categories:
            print("Predictions by category:")
            category_scores = {}
            for i in range(len(class_names)):
                category = BIOME_CATEGORIES.get(class_names[i], "unknown")
                if category not in category_scores:
                    category_scores[category] = 0.0
                category_scores[category] += float(predictions[0][i])
                
            for category, score in sorted(category_scores.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {category}: {score:.2%}")
        
        return {
            "image": os.path.basename(image_path),
            "predicted_class": predicted_class,
            "category": category if show_categories else None,
            "confidence": confidence,
            "predictions": {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
        }
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def process_directory(model, directory_path, class_names, target_height=128, target_width=128, show_categories=False):
    """Process all images in a directory"""
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    results = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    
    for filename in os.listdir(directory_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(directory_path, filename)
            result = predict_image(model, image_path, class_names, target_height, target_width, show_categories)
            if result:
                results.append(result)
            print("-" * 50)
    
    # Print summary statistics
    if results:
        class_counts = {}
        for r in results:
            cls = r["predicted_class"]
            if cls not in class_counts:
                class_counts[cls] = 0
            class_counts[cls] += 1
            
        print("\nSummary statistics:")
        print("Class distribution:")
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(results) * 100
            print(f"  - {cls}: {count} ({percentage:.1f}%)")
            
        if show_categories:
            category_counts = {}
            for r in results:
                category = r["category"]
                if category not in category_counts:
                    category_counts[category] = 0
                category_counts[category] += 1
                
            print("Category distribution:")
            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = count / len(results) * 100
                print(f"  - {category}: {count} ({percentage:.1f}%)")
    
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run inference on images using a trained model')
    parser.add_argument('--model', required=True, help='Path to the .h5 model file')
    parser.add_argument('--image', help='Path to an image file to classify')
    parser.add_argument('--dir', help='Path to a directory of images to classify')
    parser.add_argument('--height', type=int, default=128, help='Image height (default: 128)')
    parser.add_argument('--width', type=int, default=128, help='Image width (default: 128)')
    parser.add_argument('--categories', action='store_true', help='Show high-level categories for biomes')
    parser.add_argument('--output', help='Save results to a JSON file')
    args = parser.parse_args()
    
    if not args.image and not args.dir:
        parser.error("Either --image or --dir is required")
    
    # Load the model
    model = load_model(args.model)
    
    # Get class names
    dataset_path = "dataset"
    train_dir = os.path.join(dataset_path, 'train')
    
    if os.path.exists(train_dir):
        class_names = sorted(os.listdir(train_dir))
    else:
        # If dataset directory doesn't exist, ask user for class names
        print("Dataset directory not found. Please enter class names (comma-separated):")
        class_input = input()
        class_names = [name.strip() for name in class_input.split(',')]
    
    print(f"Class names: {class_names}")
    
    # Process image or directory
    results = None
    if args.image:
        result = predict_image(model, args.image, class_names, args.height, args.width, args.categories)
        results = [result] if result else []
    elif args.dir:
        results = process_directory(model, args.dir, class_names, args.height, args.width, args.categories)
        print(f"Processed {len(results)} images from {args.dir}")
    
    # Save results to a JSON file if requested
    if args.output and results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()