"""Evaluate and rank image classification models

This script evaluates multiple trained models on the test dataset and ranks them
based on various metrics like accuracy, F1 score, and inference time.

Usage:
    python evaluate_models.py
    python evaluate_models.py --models "model1.h5,model2.h5,model3.h5"
    python evaluate_models.py --dataset custom_dataset

Arguments:
    --models: Comma-separated list of model files to evaluate (default: finds all .h5 files)
    --dataset: Path to the dataset directory (default: dataset)
    --batch-size: Batch size for evaluation (default: 32)
    --detailed: Show detailed per-class metrics
    --sort-by: Metric to sort by (accuracy, f1, precision, recall, inference_time)
    --export: Export results to a CSV file
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define the biome category mappings
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
}

def find_model_files(directory="."):
    """Find all .h5 model files in the current directory"""
    model_files = []
    for file in os.listdir(directory):
        if file.endswith(".h5"):
            model_files.append(file)
    return model_files

def load_test_dataset(dataset_path="dataset", batch_size=32, img_height=128, img_width=128):
    """Load the test dataset"""
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
    
    return test_dataset, all_images, all_labels, class_names

def evaluate_model(model, model_name, images, labels, class_names, batch_size=32):
    """Evaluate a single model and compute performance metrics"""
    # Measure inference time for a single batch (batch_size images)
    single_batch = images[:batch_size]
    start_time = time.time()
    _ = model.predict(single_batch)
    end_time = time.time()
    inference_time_batch = end_time - start_time
    inference_time_per_image = inference_time_batch / len(single_batch)
    
    # Get predictions for all images
    start_time = time.time()
    predictions = model.predict(images)
    end_time = time.time()
    total_inference_time = end_time - start_time
    
    # Get predicted class indices
    pred_indices = np.argmax(predictions, axis=1)
    true_indices = np.argmax(labels, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(pred_indices == true_indices)
    
    # Generate classification report
    report = classification_report(true_indices, pred_indices, target_names=class_names, output_dict=True)
    
    # Calculate weighted averages
    weighted_precision = report['weighted avg']['precision']
    weighted_recall = report['weighted avg']['recall']
    weighted_f1 = report['weighted avg']['f1-score']
    
    # Calculate high-level category accuracy if using detailed biomes
    category_accuracy = None
    if len(set(true_indices)) > 3 and all(cls in BIOME_CATEGORIES for cls in class_names):
        # Map class names to categories
        true_categories = [BIOME_CATEGORIES.get(class_names[idx], "unknown") for idx in true_indices]
        pred_categories = [BIOME_CATEGORIES.get(class_names[idx], "unknown") for idx in pred_indices]
        category_accuracy = np.mean(np.array(true_categories) == np.array(pred_categories))
    
    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "category_accuracy": category_accuracy,
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1_score": weighted_f1,
        "inference_time_batch": inference_time_batch,
        "inference_time_per_image": inference_time_per_image,
        "total_inference_time": total_inference_time,
        "num_images": len(images),
        "report": report,
        "true_indices": true_indices,
        "pred_indices": pred_indices,
        "class_names": class_names
    }

def print_model_evaluation(eval_results, detailed=False):
    """Print evaluation results for a single model"""
    print("\n" + "=" * 50)
    print(f"Model: {eval_results['model_name']}")
    print(f"Overall Accuracy: {eval_results['accuracy']:.4f}")
    
    if eval_results['category_accuracy'] is not None:
        print(f"Category Accuracy: {eval_results['category_accuracy']:.4f}")
    
    print(f"Weighted Precision: {eval_results['precision']:.4f}")
    print(f"Weighted Recall: {eval_results['recall']:.4f}")
    print(f"Weighted F1 Score: {eval_results['f1_score']:.4f}")
    print(f"Inference Time (batch of {eval_results['num_images']} images): {eval_results['inference_time_batch']:.4f} seconds")
    print(f"Inference Time (per image): {eval_results['inference_time_per_image'] * 1000:.2f} ms")
    
    if detailed:
        print("\nPer-class Performance:")
        for class_name in eval_results['class_names']:
            if class_name in eval_results['report']:
                cls_report = eval_results['report'][class_name]
                print(f"  {class_name} - Precision: {cls_report['precision']:.4f}, Recall: {cls_report['recall']:.4f}, F1: {cls_report['f1-score']:.4f}")
    
    # Calculate and print confusion matrix
    cm = confusion_matrix(eval_results['true_indices'], eval_results['pred_indices'])
    correct_predictions = np.sum(np.diag(cm))
    total_predictions = np.sum(cm)
    print(f"\nCorrect Predictions: {correct_predictions}/{total_predictions}")

def plot_confusion_matrix(eval_results, save_path=None):
    """Plot confusion matrix for the model"""
    cm = confusion_matrix(eval_results['true_indices'], eval_results['pred_indices'])
    class_names = eval_results['class_names']
    
    plt.figure(figsize=(12, 10))
    plt.title(f"Confusion Matrix - {eval_results['model_name']}")
    
    # Normalize confusion matrix for better visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

def plot_model_comparison(eval_results_list, save_path=None):
    """Plot a single comprehensive graph comparing models by accuracy, precision, and recall"""
    if not eval_results_list:
        return
    
    # Extract model names and metrics
    model_names = [os.path.splitext(result['model_name'])[0] for result in eval_results_list]
    accuracy_scores = [result['accuracy'] for result in eval_results_list]
    precision_scores = [result['precision'] for result in eval_results_list]
    recall_scores = [result['recall'] for result in eval_results_list]
    f1_scores = [result['f1_score'] for result in eval_results_list]
    
    # Create figure and axes
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
    
    # Set position of bar on X axis
    x = np.arange(len(model_names))
    width = 0.2
    
    # Make the plot
    rects1 = ax.bar(x - width*1.5, accuracy_scores, width, label='Accuracy', color='#1f77b4')
    rects2 = ax.bar(x - width/2, precision_scores, width, label='Precision', color='#ff7f0e')
    rects3 = ax.bar(x + width/2, recall_scores, width, label='Recall', color='#2ca02c')
    rects4 = ax.bar(x + width*1.5, f1_scores, width, label='F1 Score', color='#d62728')
    
    # Add labels, title and legend
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=0)
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    
    # Add a second axis for inference time
    ax2 = ax.twinx()
    inference_times = [result['inference_time_per_image'] * 1000 for result in eval_results_list]  # in ms
    line = ax2.plot(x, inference_times, 'o-', color='#9467bd', label='Inference Time (ms)')
    
    # Set y-axis properties for inference time
    max_time = max(inference_times) * 1.2
    ax2.set_ylim(0, max_time)
    ax2.set_ylabel('Inference Time (ms/image)', color='#9467bd', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#9467bd')
    
    # Add labels for inference time
    for i, time in enumerate(inference_times):
        ax2.annotate(f'{time:.2f}ms',
                    xy=(x[i], time),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, color='#9467bd')
    
    # Add a legend for inference time
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Model comparison chart saved to {save_path}")
    else:
        plt.show()

def rank_models(eval_results_list, sort_by='accuracy'):
    """Rank models based on specified metric"""
    # Define sorting key based on specified metric
    sort_keys = {
        'accuracy': lambda x: -x['accuracy'],  # Negative for descending order
        'category_accuracy': lambda x: -x['category_accuracy'] if x['category_accuracy'] is not None else -float('inf'),
        'f1': lambda x: -x['f1_score'],
        'precision': lambda x: -x['precision'],
        'recall': lambda x: -x['recall'],
        'inference_time': lambda x: x['inference_time_per_image']  # Ascending order for time
    }
    
    if sort_by not in sort_keys:
        print(f"Warning: Unknown sorting metric '{sort_by}'. Using 'accuracy' instead.")
        sort_by = 'accuracy'
    
    # Sort models based on the specified metric
    sorted_results = sorted(eval_results_list, key=sort_keys[sort_by])
    
    # Print ranking table
    print("\n" + "=" * 80)
    print(f"Model Rankings (sorted by {sort_by}):")
    print("=" * 80)
    print(f"{'Rank':^5} | {'Model Name':^20} | {'Accuracy':^10} | {'F1 Score':^10} | {'Inference Time (ms)':^20}")
    print("-" * 80)
    
    for i, result in enumerate(sorted_results, 1):
        print(f"{i:^5} | {result['model_name']:^20} | {result['accuracy']:.4f} | {result['f1_score']:.4f} | {result['inference_time_per_image'] * 1000:.2f}")
    
    print("=" * 80)
    
    return sorted_results

def export_results_to_csv(eval_results_list, filename="model_evaluation_results.csv"):
    """Export evaluation results to a CSV file"""
    data = []
    
    for result in eval_results_list:
        row = {
            "Model Name": result["model_name"],
            "Accuracy": result["accuracy"],
            "Category Accuracy": result["category_accuracy"] if result["category_accuracy"] is not None else "N/A",
            "Precision": result["precision"],
            "Recall": result["recall"],
            "F1 Score": result["f1_score"],
            "Inference Time (ms/image)": result["inference_time_per_image"] * 1000,
            "Total Inference Time (s)": result["total_inference_time"],
            "Number of Test Images": result["num_images"]
        }
        
        # Add per-class metrics
        for class_name in result["class_names"]:
            if class_name in result["report"]:
                cls_report = result["report"][class_name]
                row[f"{class_name}_precision"] = cls_report["precision"]
                row[f"{class_name}_recall"] = cls_report["recall"]
                row[f"{class_name}_f1"] = cls_report["f1-score"]
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Results exported to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate and rank image classification models")
    parser.add_argument("--models", help="Comma-separated list of model files to evaluate")
    parser.add_argument("--dataset", default="dataset", help="Path to the dataset directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--detailed", action="store_true", help="Show detailed per-class metrics")
    parser.add_argument("--sort-by", default="accuracy", 
                        choices=["accuracy", "category_accuracy", "f1", "precision", "recall", "inference_time"],
                        help="Metric to sort models by")
    parser.add_argument("--export", help="Export results to a CSV file with the specified name")
    parser.add_argument("--plot-cm", action="store_true", help="Plot confusion matrices")
    parser.add_argument("--chart", help="Save the performance comparison chart to the specified file")
    parser.add_argument("--no-chart", action="store_true", help="Don't generate the comparison chart")
    
    args = parser.parse_args()
    
    # Find model files
    if args.models:
        model_files = args.models.split(",")
    else:
        model_files = find_model_files()
    
    if not model_files:
        print("No model files found. Please specify model files with --models.")
        return
    
    print(f"Found {len(model_files)} model files: {', '.join(model_files)}")
    
    # Load test dataset
    test_dataset, test_images, test_labels, class_names = load_test_dataset(
        dataset_path=args.dataset,
        batch_size=args.batch_size
    )
    
    print(f"Loaded test dataset with {len(test_images)} images and {len(class_names)} classes")
    print(f"Class names: {class_names}")
    
    # Evaluate each model
    eval_results_list = []
    
    for model_file in model_files:
        try:
            print(f"Loading and evaluating model: {model_file}")
            model = keras.models.load_model(model_file)
            
            # Evaluate the model
            eval_results = evaluate_model(
                model, 
                model_file, 
                test_images, 
                test_labels, 
                class_names,
                batch_size=args.batch_size
            )
            
            # Print evaluation results
            print_model_evaluation(eval_results, detailed=args.detailed)
            
            # Plot confusion matrix if requested
            if args.plot_cm:
                plot_confusion_matrix(
                    eval_results, 
                    save_path=f"confusion_matrix_{os.path.splitext(model_file)[0]}.png"
                )
            
            eval_results_list.append(eval_results)
            
        except Exception as e:
            print(f"Error evaluating model {model_file}: {e}")
    
    # Rank models
    ranked_models = rank_models(eval_results_list, sort_by=args.sort_by)
    
    # Generate comparison chart
    if len(eval_results_list) >= 2 and not args.no_chart:
        chart_path = args.chart if args.chart else None
        plot_model_comparison(eval_results_list, save_path=chart_path)
    
    # Export results if requested
    if args.export:
        export_filename = args.export
        if not export_filename.endswith(".csv"):
            export_filename += ".csv"
        export_results_to_csv(eval_results_list, filename=export_filename)

if __name__ == "__main__":
    # Check if this is running in a notebook environment
    try:
        # This will only be defined in IPython/Jupyter environments
        get_ipython()
        IN_NOTEBOOK = True
    except NameError:
        IN_NOTEBOOK = False
    
    # Call main only if not in a notebook
    if not IN_NOTEBOOK:
        main()
    # For notebooks, the functions will be called directly rather than through main()