"""Run inference on images using trained image classification models

This script loads a trained model from a .h5 file and performs inference on 
individual images or a directory of images.

Usage:
    python inference.py --model MODEL_FILE --image IMAGE_PATH
    python inference.py --model MODEL_FILE --dir DIRECTORY_PATH

Arguments:
    --model: Path to the .h5 model file
    --image: Path to an image file to classify
    --dir: Path to a directory of images to classify
    --height: Image height (default: 128)
    --width: Image width (default: 128)
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image

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

def predict_image(model, image_path, class_names, target_height=128, target_width=128):
    """Run inference on a single image"""
    try:
        # Preprocess the image
        img_array = preprocess_image(image_path, target_height, target_width)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        # Display results
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Predicted: {predicted_class} (Confidence: {confidence:.2%})")
        print("Top predictions:")
        
        # Get top 3 predictions or all if fewer than 3 classes
        top_n = min(3, len(class_names))
        top_indices = np.argsort(predictions[0])[-top_n:][::-1]
        for i in top_indices:
            print(f"  - {class_names[i]}: {predictions[0][i]:.2%}")
        
        return {
            "image": os.path.basename(image_path),
            "predicted_class": predicted_class,
            "confidence": confidence,
            "predictions": {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
        }
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def process_directory(model, directory_path, class_names, target_height=128, target_width=128):
    """Process all images in a directory"""
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    results = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    
    for filename in os.listdir(directory_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(directory_path, filename)
            result = predict_image(model, image_path, class_names, target_height, target_width)
            if result:
                results.append(result)
            print("-" * 50)
    
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run inference on images using a trained model')
    parser.add_argument('--model', required=True, help='Path to the .h5 model file')
    parser.add_argument('--image', help='Path to an image file to classify')
    parser.add_argument('--dir', help='Path to a directory of images to classify')
    parser.add_argument('--height', type=int, default=128, help='Image height (default: 128)')
    parser.add_argument('--width', type=int, default=128, help='Image width (default: 128)')
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
    if args.image:
        predict_image(model, args.image, class_names, args.height, args.width)
    elif args.dir:
        results = process_directory(model, args.dir, class_names, args.height, args.width)
        print(f"Processed {len(results)} images from {args.dir}")

if __name__ == "__main__":
    main()