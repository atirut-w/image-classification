#!/usr/bin/env python3
"""
Create a comprehensive Jupyter notebook from the image classification project.

This script combines all the functionality from the different Python files
(process_dataset.py, train.py, inference.py, evaluate_models.py) into a single
organized Jupyter notebook.

Usage:
    python create_notebook.py [--output OUTPUT_FILE]
"""

import os
import argparse
import nbformat as nbf
import inspect
import re

def extract_imports(file_path):
    """Extract import statements from a Python file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Use regex to find import statements
    imports = re.findall(r'^(?:import|from)\s+[^\n]+', content, re.MULTILINE)
    return imports

def extract_docstring(file_path):
    """Extract the docstring from a Python file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Use regex to find triple-quoted docstrings at the beginning of the file
    matches = re.match(r'^"""(.*?)"""', content, re.DOTALL)
    if matches:
        return matches.group(1).strip()
    return ""

def extract_functions(file_path):
    """Extract function definitions from a Python file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all function definitions with their content
    pattern = r'(def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*:(?:\s*"""(?:.*?)""")?(?:(?!\ndef\s+).)*)'
    functions = re.findall(pattern, content, re.DOTALL)
    return functions

def create_notebook():
    """Create a comprehensive Jupyter notebook from project files"""
    # Create a new notebook
    nb = nbf.v4.new_notebook()
    
    # Add a title cell
    nb.cells.append(nbf.v4.new_markdown_cell("""# Image Classification Project
    
This notebook contains a comprehensive implementation of the image classification project, including:
    
1. Dataset Processing - Preparing the dataset for training and testing
2. Model Training - Training multiple neural network architectures (CNN, deeper CNN, ANN)
3. Inference - Running predictions on new images
4. Model Evaluation - Comparing model performance and visualizing results

All functionality has been combined from the original Python scripts into this single notebook.
"""))
    
    # Add requirements cell
    nb.cells.append(nbf.v4.new_markdown_cell("## Requirements"))
    with open("requirements.txt", "r") as f:
        requirements = f.read()
    nb.cells.append(nbf.v4.new_code_cell(f"""# Install required packages
# Uncomment to install packages
# !pip install -r requirements.txt

# Requirements:
'''
{requirements}
'''"""))
    
    # Section 1: Imports and Setup
    nb.cells.append(nbf.v4.new_markdown_cell("## 1. Imports and Setup"))
    
    # Collect all imports from project files
    all_imports = set()
    for file_name in ["process_dataset.py", "train.py", "inference.py", "evaluate_models.py"]:
        if os.path.exists(file_name):
            imports = extract_imports(file_name)
            all_imports.update(imports)
    
    imports_cell = "\n".join(sorted(list(all_imports)))
    nb.cells.append(nbf.v4.new_code_cell(imports_cell + "\n\n# Set up parameters\nBATCH_SIZE = 32\nEPOCHS = 15\nIMG_HEIGHT = 128\nIMG_WIDTH = 128\nRANDOM_SEED = 42"))
    
    # Section 2: Dataset Processing
    nb.cells.append(nbf.v4.new_markdown_cell("## 2. Dataset Processing"))
    if os.path.exists("process_dataset.py"):
        docstring = extract_docstring("process_dataset.py")
        nb.cells.append(nbf.v4.new_markdown_cell(f"### Dataset Processing\n\n{docstring}"))
        
        functions = extract_functions("process_dataset.py")
        for func in functions:
            if "def process_dataset" in func:
                nb.cells.append(nbf.v4.new_code_cell(func))

    # Add a cell for Google Colab data download
    nb.cells.append(nbf.v4.new_markdown_cell("### Google Colab Setup"))
    nb.cells.append(nbf.v4.new_code_cell("""# Check if running in Colab
import sys
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    # Mount Google Drive to access data
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Install required packages
    !pip install -q scikit-learn pandas matplotlib seaborn
    
    # Download Nether comparison dataset
    # Option 1: If you've uploaded it to Drive
    # !cp /content/drive/MyDrive/archive.zip /content/
    
    # Option 2: Download directly (use actual URL when available)
    # !wget -q https://example.com/archive.zip -O archive.zip
    
    # Option 3: Upload via Colab's file uploader
    from google.colab import files
    print("Please upload the dataset archive.zip file:")
    uploaded = files.upload()
    
    print("Colab setup complete!")
    
    # Verify the dataset exists
    import os
    if os.path.exists('archive.zip'):
        print("Dataset archive found!")
        !ls -lh archive.zip
    else:
        print("WARNING: Dataset archive not found. Please upload it manually.")
"""))
    
    # Add a cell for running the dataset processing
    nb.cells.append(nbf.v4.new_code_cell("""# Process the dataset
data_source = "archive.zip"    # Path to the dataset archive
output_dir = "dataset"         # Path to the processed dataset
detailed = True                # Use detailed biome labels

# Process the dataset
# For Colab compatibility - call function directly instead of using command-line args
process_dataset(
    data_source=data_source,
    output_dir=output_dir,
    test_size=0.2,
    random_state=RANDOM_SEED,
    detailed=detailed
)"""))
    
    # Section 3: Model Definitions
    nb.cells.append(nbf.v4.new_markdown_cell("## 3. Model Definitions"))
    if os.path.exists("train.py"):
        # Extract and add model definitions
        with open("train.py", "r") as f:
            content = f.read()
        
        # Extract and add the parameters section
        # First add the parameters that the functions need
        parameters_code = """# Parameters needed for dataset loading and model training
BATCH_SIZE = 32   # Smaller batch size for better learning with small datasets
EPOCHS = 15       # Epochs for training
IMG_HEIGHT = 128  # Image height
IMG_WIDTH = 128   # Image width
DATASET_PATH = "dataset"
USE_SUBSET = False # Use all available data
SUBSET_SIZE = 0.2  # Use 20% of the dataset if USE_SUBSET is True
RANDOM_SEED = 42   # Random seed for reproducibility
"""
        nb.cells.append(nbf.v4.new_code_cell(parameters_code))
        
        # Extract data augmentation code
        augmentation_pattern = r'# Data augmentation for training\ndata_augmentation = .*?^\)'
        augmentation_matches = re.search(augmentation_pattern, content, re.DOTALL | re.MULTILINE)
        augmentation_code = ''
        if augmentation_matches:
            augmentation_code = augmentation_matches.group(0)
            nb.cells.append(nbf.v4.new_code_cell(augmentation_code))
        
        # Extract dataset preparation function - include data_augmentation in the same cell if not found
        prepare_pattern = r'def prepare_dataset.*?^\s*return dataset'
        prepare_matches = re.search(prepare_pattern, content, re.DOTALL | re.MULTILINE)
        if prepare_matches:
            prepare_code = prepare_matches.group(0)
            # If augmentation code wasn't found earlier, include a default version
            if not augmentation_matches:
                augmentation_code = """# Data augmentation for training
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomZoom(0.2),
])"""
                prepare_code = augmentation_code + "\n\n" + prepare_code
            nb.cells.append(nbf.v4.new_code_cell(prepare_code))
        
        # Add model definitions
        nb.cells.append(nbf.v4.new_markdown_cell("### CNN Model"))
        cnn_pattern = r'def create_cnn_model.*?^\s*return model'
        cnn_matches = re.search(cnn_pattern, content, re.DOTALL | re.MULTILINE)
        if cnn_matches:
            cnn_code = cnn_matches.group(0)
            nb.cells.append(nbf.v4.new_code_cell(cnn_code))
        
        nb.cells.append(nbf.v4.new_markdown_cell("### Deeper CNN Model"))
        deeper_pattern = r'def create_deeper_cnn_model.*?^\s*return model'
        deeper_matches = re.search(deeper_pattern, content, re.DOTALL | re.MULTILINE)
        if deeper_matches:
            deeper_code = deeper_matches.group(0)
            nb.cells.append(nbf.v4.new_code_cell(deeper_code))
        
        nb.cells.append(nbf.v4.new_markdown_cell("### ANN Model"))
        ann_pattern = r'def create_ann_model.*?^\s*return model'
        ann_matches = re.search(ann_pattern, content, re.DOTALL | re.MULTILINE)
        if ann_matches:
            ann_code = ann_matches.group(0)
            nb.cells.append(nbf.v4.new_code_cell(ann_code))
        
        # Add training function
        nb.cells.append(nbf.v4.new_markdown_cell("### Training Function"))
        train_pattern = r'def train_and_save_model.*?^\s*return history'
        train_matches = re.search(train_pattern, content, re.DOTALL | re.MULTILINE)
        if train_matches:
            train_code = train_matches.group(0)
            nb.cells.append(nbf.v4.new_code_cell(train_code))
    
    # Section 4: Model Training
    nb.cells.append(nbf.v4.new_markdown_cell("## 4. Model Training"))
    nb.cells.append(nbf.v4.new_code_cell("""# Ensure data augmentation is defined
if 'data_augmentation' not in globals():
    print("Defining data augmentation...")
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.2),
        keras.layers.RandomZoom(0.2),
    ])

# Load the dataset
DATASET_PATH = "dataset"

try:
    # Get class names
    train_dir = os.path.join(DATASET_PATH, 'train')
    class_names = sorted(os.listdir(train_dir))
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Load datasets
    train_dataset = prepare_dataset('train', is_training=True)
    test_dataset = prepare_dataset('test')
    
except ValueError as e:
    print(f"Error: {e}")
    print("Make sure you've processed the dataset first!")

# Create and train the models
MODEL_NAME_SUFFIX = "biomes"  # Suffix for model names

# Train Simple CNN
print("\\nTraining Simple CNN...")
cnn_model = create_cnn_model()
cnn_history = train_and_save_model(cnn_model, f"simple_cnn_{MODEL_NAME_SUFFIX}")

# Train Deeper CNN
print("\\nTraining Deeper CNN...")
deeper_cnn_model = create_deeper_cnn_model()
deeper_cnn_history = train_and_save_model(deeper_cnn_model, f"deeper_cnn_{MODEL_NAME_SUFFIX}")

# Train ANN
print("\\nTraining ANN...")
ann_model = create_ann_model()
ann_history = train_and_save_model(ann_model, f"ann_{MODEL_NAME_SUFFIX}")

print("\\nTraining complete!")"""))
    
    # Add a cell to plot training history
    nb.cells.append(nbf.v4.new_code_cell("""# Plot training history
def plot_training_history(history, title):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{title} - Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'{title} - Loss')
    plt.show()

# Plot training history for each model
plot_training_history(cnn_history, "Simple CNN")
plot_training_history(deeper_cnn_history, "Deeper CNN")
plot_training_history(ann_history, "ANN")"""))
    
    # Section 5: Model Inference
    nb.cells.append(nbf.v4.new_markdown_cell("## 5. Model Inference"))
    if os.path.exists("inference.py"):
        # Add biome category mapping
        with open("inference.py", "r") as f:
            content = f.read()
        
        biome_mapping_pattern = r'# Define category mappings for biomes\nBIOME_CATEGORIES = \{.*?\}'
        biome_mapping_matches = re.search(biome_mapping_pattern, content, re.DOTALL)
        if biome_mapping_matches:
            biome_mapping_code = biome_mapping_matches.group(0)
            nb.cells.append(nbf.v4.new_code_cell(biome_mapping_code))
        
        # Add inference functions
        functions = extract_functions("inference.py")
        for function_code in functions:
            if "def load_model" in function_code or "def preprocess_image" in function_code or "def predict_image" in function_code:
                nb.cells.append(nbf.v4.new_code_cell(function_code))
    
    # Add a cell for running inference
    nb.cells.append(nbf.v4.new_code_cell("""# Run inference on a sample image
def run_inference_demo(model_path, image_path):
    # Load the model
    model = load_model(model_path)
    
    # Get class names
    train_dir = os.path.join(DATASET_PATH, 'train')
    class_names = sorted(os.listdir(train_dir))
    
    # Run prediction
    print(f"Running inference on {image_path} with model {model_path}")
    result = predict_image(model, image_path, class_names, 
                         target_height=IMG_HEIGHT, target_width=IMG_WIDTH, 
                         show_categories=True)
    
    # Display the image
    img = plt.imread(image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"Predicted: {result['predicted_class']} ({result['confidence']:.2%})")
    plt.axis('off')
    plt.show()
    
    return result

# Example usage (uncomment and modify to run)
# model_path = "deeper_cnn_biomes.h5"
# image_path = "nether_comparison_data/new_nether_sc/crimson_forest/2021-06-30_16.09.35.png"
# result = run_inference_demo(model_path, image_path)"""))
    
    # Section 6: Model Evaluation
    nb.cells.append(nbf.v4.new_markdown_cell("## 6. Model Evaluation"))
    if os.path.exists("evaluate_models.py"):
        # First add the biome categories mapping
        with open("evaluate_models.py", "r") as f:
            content = f.read()
        
        # Extract BIOME_CATEGORIES
        biome_mapping_pattern = r'# Define.*category mappings.*\nBIOME_CATEGORIES = \{.*?\}'
        biome_mapping_matches = re.search(biome_mapping_pattern, content, re.DOTALL)
        
        if biome_mapping_matches:
            biome_mapping_code = biome_mapping_matches.group(0)
            nb.cells.append(nbf.v4.new_code_cell(biome_mapping_code))
        else:
            # Provide a default mapping if not found
            nb.cells.append(nbf.v4.new_code_cell("""# Define the biome category mappings
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
    "old_nether_sc": "old_nether", 
    "overworld": "overworld"
}"""))
        
        # Extract key evaluation functions
        functions = extract_functions("evaluate_models.py")
        evaluation_functions = []
        
        # Then, ensure we add the load_test_dataset function
        load_test_dataset_function = next((fn for fn in functions if "def load_test_dataset" in fn), None)
        if load_test_dataset_function:
            evaluation_functions.append(load_test_dataset_function)
        else:
            # If function not found, provide a default implementation
            evaluation_functions.append("""def load_test_dataset(dataset_path="dataset", batch_size=32, img_height=128, img_width=128):
    \"\"\"Load the test dataset\"\"\"
    test_dataset = keras.utils.image_dataset_from_directory(
        os.path.join(dataset_path, 'test'),
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    )
    
    # Rescale pixel values
    test_dataset = test_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    
    # Get class names
    class_names = sorted(os.listdir(os.path.join(dataset_path, 'test')))
    
    # Extract all images and labels for evaluation
    all_images = []
    all_labels = []
    
    for images, labels in test_dataset:
        all_images.append(images.numpy())
        all_labels.append(labels.numpy())
    
    if all_images:
        all_images = np.vstack(all_images)
        all_labels = np.vstack(all_labels)
    
    return test_dataset, all_images, all_labels, class_names""")
        
        # Then add the other evaluation functions
        for function_code in functions:
            if "def evaluate_model" in function_code or "def plot_confusion_matrix" in function_code or "def plot_model_comparison" in function_code:
                evaluation_functions.append(function_code)
        
        # Add evaluation functions as a single cell
        nb.cells.append(nbf.v4.new_code_cell("\n\n".join(evaluation_functions)))
    
    # Add a cell for running evaluation
    nb.cells.append(nbf.v4.new_code_cell("""# Evaluate all models
def evaluate_all_models():
    # Find all .h5 model files
    model_files = [f for f in os.listdir('.') if f.endswith('.h5')]
    
    if not model_files:
        print("No model files found!")
        return
    
    print(f"Found {len(model_files)} model files: {', '.join(model_files)}")
    
    # Load test dataset
    test_dataset, test_images, test_labels, class_names = load_test_dataset(
        dataset_path=DATASET_PATH,
        batch_size=BATCH_SIZE
    )
    
    # Evaluate each model
    eval_results_list = []
    
    for model_file in model_files:
        try:
            print(f"Loading and evaluating model: {model_file}")
            model = load_model(model_file)
            
            # Evaluate the model
            eval_results = evaluate_model(
                model, 
                model_file, 
                test_images, 
                test_labels, 
                class_names,
                batch_size=BATCH_SIZE
            )
            
            # Print basic evaluation results
            print(f"\\nModel: {eval_results['model_name']}")
            print(f"Accuracy: {eval_results['accuracy']:.4f}")
            print(f"F1 Score: {eval_results['f1_score']:.4f}")
            print(f"Inference Time (per image): {eval_results['inference_time_per_image'] * 1000:.2f} ms")
            
            eval_results_list.append(eval_results)
            
        except Exception as e:
            print(f"Error evaluating model {model_file}: {e}")
    
    # Generate comparison chart if we have multiple models
    if len(eval_results_list) >= 2:
        plot_model_comparison(eval_results_list)
    
    return eval_results_list

# Run evaluation (uncomment to run)
# eval_results = evaluate_all_models()"""))
    
    # Add a cell for confusion matrix
    nb.cells.append(nbf.v4.new_code_cell("""# Plot confusion matrix for a specific model
def plot_model_confusion_matrix(model_file):
    # Load model
    model = load_model(model_file)
    
    # Load test dataset
    test_dataset, test_images, test_labels, class_names = load_test_dataset(
        dataset_path=DATASET_PATH,
        batch_size=BATCH_SIZE
    )
    
    # Evaluate
    eval_results = evaluate_model(
        model, 
        model_file, 
        test_images, 
        test_labels, 
        class_names,
        batch_size=BATCH_SIZE
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(eval_results)
    
    return eval_results

# Example (uncomment to run)
# plot_model_confusion_matrix("deeper_cnn_biomes.h5")"""))
    
    return nb

def main():
    parser = argparse.ArgumentParser(description="Create a Jupyter notebook from project files")
    parser.add_argument("--output", default="image_classification_notebook.ipynb", help="Output notebook filename")
    args = parser.parse_args()
    
    print(f"Creating notebook: {args.output}")
    nb = create_notebook()
    
    # Write the notebook to a file
    with open(args.output, 'w') as f:
        nbf.write(nb, f)
    
    print(f"Notebook created successfully: {args.output}")
    print(f"You can open it with: jupyter notebook {args.output}")

if __name__ == "__main__":
    main()