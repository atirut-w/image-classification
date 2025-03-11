"""Trains image classification models

This script trains three image classification models (one being an ANN) using
Keras, using images from the `dataset` directory, and saves the models as
individual `.h5` files in the current working directory.

The structure of the dataset directory must look like so:

dataset/
    train/
        label1/
            image1.png
            ...
        label2/
            image1.png
            ...
        ...
    test/
        label1/
            image1.png
            ...
        label2/
            image1.png
            ...
        ...
"""

import os
import tensorflow as tf
import time
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling, RandomFlip, RandomRotation, RandomZoom
from keras.optimizers import Adam
from keras.utils import image_dataset_from_directory

# Parameters - extremely optimized for laptop CPU
BATCH_SIZE = 32   # Smaller batch size for better learning with small datasets
EPOCHS = 15       # Increased epochs for more detailed classes
IMG_HEIGHT = 128  # Very small image size
IMG_WIDTH = 128   # Very small image size
DATASET_PATH = "dataset"
USE_SUBSET = False # Use all available data
SUBSET_SIZE = 0.2  # Use 20% of the dataset if USE_SUBSET is True
MODEL_NAME_SUFFIX = ""  # Optional suffix for model names

# Data augmentation for training
data_augmentation = keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.2),
])

# Load and prepare datasets
def prepare_dataset(subset_dir, is_training=False):
    dataset_dir = os.path.join(DATASET_PATH, subset_dir)
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory {dataset_dir} does not exist")
    
    # Set validation_split and subset parameters if using a subset of data
    if USE_SUBSET:
        dataset = image_dataset_from_directory(
            dataset_dir,
            validation_split=1-SUBSET_SIZE if is_training else SUBSET_SIZE,
            subset="training" if is_training else "validation",
            seed=42,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            label_mode='categorical',
            shuffle=is_training
        )
    else:
        dataset = image_dataset_from_directory(
            dataset_dir,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            label_mode='categorical',
            shuffle=is_training
        )
    
    # Rescale pixel values
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    
    # Apply data augmentation only for training
    if is_training:
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Use buffered prefetching
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

try:
    # Load datasets
    train_dir = os.path.join(DATASET_PATH, 'train')
    test_dir = os.path.join(DATASET_PATH, 'test')
    
    # Get class names directly from directories
    class_names = sorted(os.listdir(train_dir))
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Load datasets
    train_dataset = prepare_dataset('train', is_training=True)
    test_dataset = prepare_dataset('test')
    
except ValueError as e:
    print(f"Error: {e}")
    print("Creating sample dataset structure in 'dataset' directory...")
    # Create directory structure as per docstring instructions
    for subset in ['train', 'test']:
        for label in ['label1', 'label2']:
            os.makedirs(os.path.join(DATASET_PATH, subset, label), exist_ok=True)
    print("Directory structure created. Please add images to these directories and run the script again.")
    exit(1)

# Model 1: Simple CNN - improved architecture for CPU training
def create_cnn_model():
    model = Sequential([
        # First convolutional block - double filters
        Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        
        # Second convolutional block - double filters
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        
        # Third convolutional block - double filters
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        
        # Flatten and dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),  # Less dropout to retain more information
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Lower learning rate for better convergence
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Model 2: ResNet-style CNN with skip connections
def create_deeper_cnn_model():
    # Cannot use Sequential for a ResNet-style model with skip connections
    # We need to use the Functional API instead
    
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Initial convolution
    x = Conv2D(32, (7, 7), strides=2, padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    
    # First residual block
    residual = x
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.add([x, residual])  # Skip connection
    x = keras.layers.Activation('relu')(x)
    
    # Second residual block with projection
    residual = Conv2D(64, (1, 1), strides=2, padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = Conv2D(64, (3, 3), strides=2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.add([x, residual])  # Skip connection with projection
    x = keras.layers.Activation('relu')(x)
    
    # Third residual block with projection
    residual = Conv2D(128, (1, 1), strides=2, padding='same')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = Conv2D(128, (3, 3), strides=2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.add([x, residual])  # Skip connection with projection
    x = keras.layers.Activation('relu')(x)
    
    # Global average pooling and classification
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create and compile model
    model = keras.Model(inputs, outputs, name="resnet_style_cnn")
    
    model.compile(
        optimizer=Adam(learning_rate=0.0002),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Model 3: ANN (No convolutions) - improved architecture for CPU training
def create_ann_model():
    model = Sequential([
        Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        
        # Larger and deeper network
        Dense(512, activation='relu'),
        Dropout(0.3),
        
        Dense(256, activation='relu'),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])
    
    # Add batch normalization and regularization to improve training
    model.compile(
        optimizer=Adam(learning_rate=0.0003),  # Lower learning rate for better convergence
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Train and save models
def train_and_save_model(model, model_name):
    # Add suffix if provided
    if MODEL_NAME_SUFFIX:
        model_name = f"{model_name}_{MODEL_NAME_SUFFIX}"
    
    print(f"\nTraining {model_name}...")
    model.summary()
    
    # Time the training process
    start_time = time.time()
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=test_dataset
    )
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"{model_name} - Test accuracy: {test_acc:.4f}")
    
    # Save the model
    model.save(f'{model_name}.h5')
    print(f"Model saved as '{model_name}.h5'")
    
    return history

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train image classification models")
    parser.add_argument("--dataset", default="dataset", help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--model-suffix", default="", help="Suffix to add to model names")
    parser.add_argument("--models", default="all", choices=["all", "cnn", "deeper_cnn", "ann"], 
                        help="Which models to train (default: all)")
    
    args = parser.parse_args()
    
    # Update global parameters
    DATASET_PATH = args.dataset
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    MODEL_NAME_SUFFIX = args.model_suffix
    
    # Create and train the selected models
    if args.models in ["all", "cnn"]:
        cnn_model = create_cnn_model()
        train_and_save_model(cnn_model, "simple_cnn")
    
    if args.models in ["all", "deeper_cnn"]:
        deeper_cnn_model = create_deeper_cnn_model()
        train_and_save_model(deeper_cnn_model, "deeper_cnn")
    
    if args.models in ["all", "ann"]:
        ann_model = create_ann_model()
        train_and_save_model(ann_model, "ann")
    
    print("\nTraining complete. All models saved.")
