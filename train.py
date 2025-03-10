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
BATCH_SIZE = 128  # Very large batch size for faster training
EPOCHS = 3        # Minimal epochs
IMG_HEIGHT = 128  # Very small image size
IMG_WIDTH = 128   # Very small image size
DATASET_PATH = "dataset"
USE_SUBSET = True # Use only a subset of data for even faster training
SUBSET_SIZE = 0.2 # Use 20% of the dataset

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

# Model 1: Simple CNN - tiny architecture for CPU training
def create_cnn_model():
    model = Sequential([
        Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(2, 2),
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Model 2: Deeper CNN - tiny architecture for CPU training
def create_deeper_cnn_model():
    model = Sequential([
        Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        Conv2D(8, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Model 3: ANN (No convolutions) - tiny architecture for CPU training
def create_ann_model():
    model = Sequential([
        Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Train and save models
def train_and_save_model(model, model_name):
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

# Create and train the models
cnn_model = create_cnn_model()
train_and_save_model(cnn_model, "simple_cnn")

deeper_cnn_model = create_deeper_cnn_model()
train_and_save_model(deeper_cnn_model, "deeper_cnn")

ann_model = create_ann_model()
train_and_save_model(ann_model, "ann")

print("\nTraining complete. All models saved.")
