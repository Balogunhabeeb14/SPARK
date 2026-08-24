# Importing all the necessary libraries
import os
import csv
import json
import random
from pathlib import Path
import mlflow
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Import evaluation metrics and utilities from scikit-learn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# PATHS & DIRECTORY SETUP
# ==========================================
# Determine the base directory and project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Path to the main dataset folder
DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "ai_plastic_waste_management_training_updated"
)

# Shared split file for Round 11.
# EfficientNetB3 and ResNet50 must use the same file to ensure fair comparison.
SPLIT_MANIFEST_PATH = os.path.join(
    PROJECT_ROOT,
    "round11_grouped_split_70_20_10.csv"
)

# Directory where MLflow will save experiment tracking data
MLFLOW_DIR = os.path.join(
    PROJECT_ROOT,
    "mlruns"
)

# ==========================================
# HYPERPARAMETERS & CONFIGURATIONS
# ==========================================
MODEL_NAME = "ResNet50"

# Image processing and training variables
IMG_SIZE = (224, 224)    # Standard image size for ResNet50
BATCH_SIZE = 64          # Number of images processed at once
SEED = 123               # Random seed for reproducibility
EPOCHS = 40              # Number of times the model will see the dataset
LEARNING_RATE = 1e-4     # How fast the model learns
UNFREEZE_LAYERS = 60     # Number of layers to unfreeze in the base model for fine-tuning

# Train, validation, and test split ratios
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.20
TEST_SPLIT = 0.10

# Set to True if you want to overwrite the existing split CSV
FORCE_RECREATE_SPLIT = False

# Expected subfolders representing the plastic classes
VALID_CLASSES = [
    "HDPE Plastic",
    "LDPE Plastic",
    "PET Plastic",
    "PP Plastic",
    "PS Plastic"
]

# Accepted image file types
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png"
)

# Output paths for saving the model, history, and final metrics
MODEL_SAVE_PATH = os.path.join(
    BASE_DIR,
    "plastic_model_ResNet50_round11.keras"
)

HISTORY_PATH = os.path.join(
    BASE_DIR,
    "training_history_ResNet50_round11.csv"
)

METRICS_PATH = os.path.join(
    BASE_DIR,
    "evaluation_metrics_ResNet50_round11.csv"
)

# ==========================================
# REPRODUCIBILITY & SYSTEM CHECKS
# ==========================================
def set_seed(seed):
    """Locks random seeds to ensure training results are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(SEED)

# Verify that the dataset path and all required class folders exist
if not os.path.isdir(DATA_PATH):
    raise FileNotFoundError(
        "\nDataset folder not found:\n"
        f"{DATA_PATH}\n\n"
        "Expected this folder to contain:\n"
        "HDPE Plastic\n"
        "LDPE Plastic\n"
        "PET Plastic\n"
        "PP Plastic\n"
        "PS Plastic\n"
    )

for class_name in VALID_CLASSES:
    class_path = os.path.join(DATA_PATH, class_name)
    if not os.path.isdir(class_path):
        raise FileNotFoundError(
            f"Missing class folder:\n{class_path}"
        )

# ==========================================
# MLFLOW TRACKING SETUP
# ==========================================
# Create MLflow directory if it doesn't exist
os.makedirs(MLFLOW_DIR, exist_ok=True)

# Tell MLflow where to save run logs
mlflow.set_tracking_uri(
    Path(MLFLOW_DIR).resolve().as_uri()
)

# Group runs under a specific experiment name
mlflow.set_experiment(
    "PlastiSort_Round_11_70_20_10"
)

# ==========================================
# DATA LOADING & SPLITTING
# ==========================================
def load_dataset(root_dir):
    """
    Scans the dataset folder and groups related images together.
    Groups are based on file names (e.g., removing "_gray" to group color and grayscale versions).
    This prevents data leakage between train and test sets.
    """
    class_names = sorted([
        folder
        for folder in os.listdir(root_dir)
        if folder in VALID_CLASSES
    ])

    image_dict = {}

    # Traverse directories to find images
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(root_dir, class_name)

        for root, _, files in os.walk(class_path):
            for file_name in files:
                if file_name.lower().endswith(IMAGE_EXTENSIONS):
                    # Clean the filename to group related images
                    base_name = (
                        file_name
                        .replace("_gray", "")
                        .replace("_grey", "")
                        .split(".")[0]
                    )
                    full_path = os.path.join(root, file_name)

                    if base_name not in image_dict:
                        image_dict[base_name] = {
                            "paths": [],
                            "label": label
                        }
                    # Add path to the group
                    image_dict[base_name]["paths"].append(full_path)

    return image_dict, class_names


# Load the data mapping
image_dict, class_names = load_dataset(DATA_PATH)

def create_and_save_split():
    """
    Splits the grouped images into Train, Validation, and Test sets using stratified sampling.
    Saves the assignments to a CSV file to be reused by other models.
    """
    all_groups = list(image_dict.keys())
    all_labels = [
        image_dict[group]["label"] for group in all_groups
    ]

    # Split off the training set (70%), leaving 30% for validation/test
    train_groups, temporary_groups = train_test_split(
        all_groups,
        test_size=(VAL_SPLIT + TEST_SPLIT),
        stratify=all_labels,
        random_state=SEED
    )

    temporary_labels = [
        image_dict[group]["label"] for group in temporary_groups
    ]

    # Split the remaining 30% into validation (20%) and test (10%)
    validation_groups, test_groups = train_test_split(
        temporary_groups,
        test_size=(TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT)),
        stratify=temporary_labels,
        random_state=SEED
    )

    # Write the split data to a CSV manifest
    with open(SPLIT_MANIFEST_PATH, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "path", "label"])

        # Log Train paths
        for group in train_groups:
            for path in image_dict[group]["paths"]:
                writer.writerow(["train", path, image_dict[group]["label"]])

        # Log Validation paths
        for group in validation_groups:
            for path in image_dict[group]["paths"]:
                writer.writerow(["validation", path, image_dict[group]["label"]])

        # Log Test paths
        for group in test_groups:
            for path in image_dict[group]["paths"]:
                writer.writerow(["test", path, image_dict[group]["label"]])

    print("Created Round 11 shared split file:")
    print(SPLIT_MANIFEST_PATH)

def load_split_from_csv():
    """Reads the dataset splits back from the saved CSV manifest."""
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    with open(SPLIT_MANIFEST_PATH, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            split_name = row["split"]
            path = row["path"]
            label = int(row["label"])

            # Safety check: ensure file hasn't been moved
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"\nImage not found:\n{path}"
                )

            # Route paths to the correct lists
            if split_name == "train":
                train_paths.append(path)
                train_labels.append(label)
            elif split_name == "validation":
                val_paths.append(path)
                val_labels.append(label)
            elif split_name == "test":
                test_paths.append(path)
                test_labels.append(label)

    return (
        train_paths, train_labels,
        val_paths, val_labels,
        test_paths, test_labels
    )

# Create the CSV if it's missing or if a fresh split is forced
if not os.path.exists(SPLIT_MANIFEST_PATH) or FORCE_RECREATE_SPLIT:
    create_and_save_split()

# Extract paths and labels from the CSV
(
    train_paths_raw, train_labels_raw,
    val_paths_raw, val_labels_raw,
    test_paths_raw, test_labels_raw
) = load_split_from_csv()

# ==========================================
# CLASS WEIGHTS & DATASET SUMMARY
# ==========================================
# Compute class weights to help the model learn equally from imbalanced classes (like PP/PS)
class_weight_values = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels_raw),
    y=train_labels_raw
)

class_weight_dict = dict(enumerate(class_weight_values))

print("Class weights (balanced):")
for index, class_name in enumerate(class_names):
    print(
        f"  {class_name:<20}: "
        f"{class_weight_dict[index]:.4f}"
    )

# Print a summary of the dataset
print("\n" + "=" * 70)
print("ROUND 11 - RESNET50")
print("Dataset Summary")
print("=" * 70)

print(f"Dataset path: {DATA_PATH}")
print(f"Plastic classes: {class_names}")
print(f"Total image groups: {len(image_dict)}")

# Calculate how many total images exist per class
class_counts = {i: 0 for i in range(len(class_names))}
for group in image_dict.values():
    class_counts[group["label"]] += len(group["paths"])

print("\nImages per class:")
for index, class_name in enumerate(class_names):
    print(f"{class_name:<20}: {class_counts[index]}")

print("Grouped stratified split:")
print(f"Training images   : {len(train_paths_raw)}")
print(f"Validation images : {len(val_paths_raw)}")
print(f"Test images       : {len(test_paths_raw)}")

print("Class distribution per split:")
print(f"{'Class':<20} {'Train':>8} {'Val':>8} {'Test':>8}")
print("-" * 55)

for index, class_name in enumerate(class_names):
    print(
        f"{class_name:<20} "
        f"{train_labels_raw.count(index):>8} "
        f"{val_labels_raw.count(index):>8} "
        f"{test_labels_raw.count(index):>8}"
    )

print("=" * 70)

# ==========================================
# TENSORFLOW DATA PIPELINE
# ==========================================
def process_image(path, label):
    """
    Reads an image from disk, resizes it, and converts it to a float32 tensor.
    Note: ResNet50 mathematical preprocessing is applied later via a custom layer.
    """
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    image.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return image, label

def create_dataset(paths, labels, shuffle=False):
    """Builds a highly efficient tf.data pipeline for feeding data to the GPU."""
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        process_image, num_parallel_calls=tf.data.AUTOTUNE
    )
    # Only shuffle the training dataset
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=1000, seed=SEED,
            reshuffle_each_iteration=True
        )
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE) # Preloads next batch into memory
    return dataset

# Generate datasets for train, validation, and test
train_ds = create_dataset(train_paths_raw, train_labels_raw, shuffle=True)
train_eval_ds = create_dataset(train_paths_raw, train_labels_raw, shuffle=False)
val_ds = create_dataset(val_paths_raw, val_labels_raw, shuffle=False)
test_ds = create_dataset(test_paths_raw, test_labels_raw, shuffle=False)

# ==========================================
# MLFLOW CALLBACK
# ==========================================
class MLflowMetricsCallback(tf.keras.callbacks.Callback):
    """Custom callback to log loss and accuracy to MLflow at the end of each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        metrics = {
            f"epoch_{key}": float(value)
            for key, value in logs.items()
            if value is not None
        }
        mlflow.log_metrics(metrics, step=epoch + 1)

# ==========================================
# MODEL ARCHITECTURE
# ==========================================
# Define data augmentation to artificially create varied images (prevents overfitting)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(0.25),
    layers.RandomContrast(0.25),
    layers.GaussianNoise(0.03),
], name="data_augmentation")

# Load pre-trained ResNet50 model (excluding the top classification layers)
base_model = tf.keras.applications.ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze lower layers; only fine-tune the upper `UNFREEZE_LAYERS`
base_model.trainable = True
for layer in base_model.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False

# Custom Preprocessing Layer specifically for ResNet50
@tf.keras.utils.register_keras_serializable(
    package="PlastiSort"
)
class ResNet50Preprocess(layers.Layer):
    """
    Applies official ResNet50 preprocessing (e.g., centering RGB values) after augmentation.
    """
    def call(self, inputs):
        return tf.keras.applications.resnet50.preprocess_input(
            inputs
        )

# Construct the full sequential model
model = models.Sequential([
    data_augmentation,                                    # Apply random transformations
    ResNet50Preprocess(name="resnet50_preprocess"),       # Preprocess inputs for ResNet
    base_model,                                           # Feature extraction backbone
    layers.GlobalAveragePooling2D(),                      # Flatten spatial dimensions
    layers.Dropout(0.5),                                  # Dropout to prevent overfitting
    layers.Dense(len(class_names), activation="softmax")  # Output layer (5 classes)
], name="round11_ResNet50")

# Compile the model with optimizer, loss function, and metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Build model to finalize input shapes
model.build(input_shape=(None, IMG_SIZE[0], IMG_SIZE[1], 3))

# Save a text summary of the architecture to disk
MODEL_SUMMARY_PATH = os.path.join(
    BASE_DIR,
    "model_summary_ResNet50_round11.txt"
)

with open(MODEL_SUMMARY_PATH, "w") as file:
    model.summary(print_fn=lambda line: file.write(line + "\n"))

print("\n" + "=" * 70)
print("MODEL ARCHITECTURE - RESNET50")
print("=" * 70)
model.summary()

# ==========================================
# EARLY STOPPING CALLBACK
# ==========================================
# Stop training early if the validation accuracy hasn't improved in 10 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    min_delta=0.005,
    restore_best_weights=True,
    verbose=1
)

# ==========================================
# EVALUATION FUNCTION
# ==========================================
def evaluate_dataset(dataset, split_name, paths):
    """
    Evaluates the model on a given dataset split (Train/Val/Test).
    Calculates metrics, creates a confusion matrix, and saves everything to CSV/txt files.
    """
    true_labels = []
    for _, label_batch in dataset:
        true_labels.extend(label_batch.numpy())
    true_labels = np.array(true_labels)

    # Get model predictions
    probabilities = model.predict(dataset, verbose=0)
    predicted_classes = np.argmax(probabilities, axis=1)
    confidence_scores = np.max(probabilities, axis=1)

    # Calculate standard metrics
    acc = accuracy_score(true_labels, predicted_classes)
    prec = precision_score(
        true_labels, predicted_classes,
        average="weighted", zero_division=0
    )
    rec = recall_score(
        true_labels, predicted_classes,
        average="weighted", zero_division=0
    )
    f1 = f1_score(
        true_labels, predicted_classes,
        average="weighted", zero_division=0
    )
    f1_per_class = f1_score(
        true_labels, predicted_classes,
        labels=list(range(len(class_names))),
        average=None, zero_division=0
    )
    
    # Generate detailed reports
    report = classification_report(
        true_labels, predicted_classes,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4, zero_division=0
    )
    cm = confusion_matrix(
        true_labels, predicted_classes,
        labels=list(range(len(class_names)))
    )

    # Define paths for saving outputs
    report_path = os.path.join(
        BASE_DIR,
        f"classification_report_ResNet50_round11_{split_name}.txt"
    )
    cm_path = os.path.join(
        BASE_DIR,
        f"confusion_matrix_ResNet50_round11_{split_name}.csv"
    )
    prediction_path = os.path.join(
        BASE_DIR,
        f"prediction_results_ResNet50_round11_{split_name}.csv"
    )

    # Save classification report to txt
    with open(report_path, "w") as file:
        file.write(report)

    # Save confusion matrix to CSV
    with open(cm_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["True / Predicted"] + class_names)
        for index, row in enumerate(cm):
            writer.writerow([class_names[index]] + row.tolist())

    # Save granular per-image predictions to CSV
    with open(prediction_path, "w", newline="") as file:
        writer = csv.writer(file)
        header = [
            "image_path", "true_label", "true_class",
            "predicted_label", "predicted_class", "confidence"
        ]
        for class_name in class_names:
            header.append(f"probability_{class_name.replace(' ', '_')}")
        writer.writerow(header)

        for index in range(len(paths)):
            row = [
                paths[index],
                int(true_labels[index]),
                class_names[true_labels[index]],
                int(predicted_classes[index]),
                class_names[predicted_classes[index]],
                float(confidence_scores[index])
            ]
            row.extend(probabilities[index].tolist())
            writer.writerow(row)

    # Print summary to terminal
    print("\n" + "=" * 70)
    print(f"{split_name.upper()} EVALUATION - {MODEL_NAME}")
    print("=" * 70)
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    # Return metrics as a dictionary for later logging
    return {
        "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1,
        "f1_per_class": f1_per_class,
        "report_path": report_path,
        "cm_path": cm_path,
        "prediction_path": prediction_path
    }

# ==========================================
# TRAINING EXECUTION & MLFLOW LOGGING
# ==========================================
# Initialize an MLflow run to log all activities
with mlflow.start_run(run_name="Round_11_ResNet50_70_20_10"):

    # Log identifying tags
    mlflow.set_tags({
        "project": "PlastiSort AI",
        "round": "Round 11",
        "model_name": MODEL_NAME,
        "split": "70/20/10",
        "split_type": "grouped_stratified"
    })

    # Log model hyperparameters
    mlflow.log_params({
        "model_name": MODEL_NAME,
        "dataset_path": DATA_PATH,
        "train_split": TRAIN_SPLIT,
        "validation_split": VAL_SPLIT,
        "test_split": TEST_SPLIT,
        "image_size": "224x224",
        "batch_size": BATCH_SIZE,
        "epochs_requested": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "unfreeze_layers": UNFREEZE_LAYERS,
        "dropout": 0.5,
        "augmentation": (
            "RandomFlip(horizontal), "
            "RandomRotation(0.3), "
            "RandomZoom(0.2), "
            "RandomTranslation(0.1, 0.1), "
            "RandomBrightness(0.25), "
            "RandomContrast(0.25), "
            "GaussianNoise(0.03)"
        ),
        "class_weighting": "balanced",
        "train_images": len(train_paths_raw),
        "validation_images": len(val_paths_raw),
        "test_images": len(test_paths_raw)
    })

    print("\n" + "=" * 70)
    print("TRAINING - ROUND 11 RESNET50")
    print("=" * 70)

    # Begin the training process!
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, MLflowMetricsCallback()],
        verbose=1
    )

    # Save the trained model file
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to:\n{MODEL_SAVE_PATH}")

    # Export epoch-by-epoch loss and accuracy metrics to a CSV
    with open(HISTORY_PATH, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch", "train_loss", "train_acc", "val_loss", "val_acc"
        ])
        history_data = history.history
        for index in range(len(history_data["loss"])):
            writer.writerow([
                index + 1,
                history_data["loss"][index],
                history_data["accuracy"][index],
                history_data["val_loss"][index],
                history_data["val_accuracy"][index]
            ])

    print(f"Training history saved to:\n{HISTORY_PATH}")

    # Evaluate the final model on all data splits
    train_metrics = evaluate_dataset(train_eval_ds, "train", train_paths_raw)
    val_metrics = evaluate_dataset(val_ds, "validation", val_paths_raw)
    test_metrics = evaluate_dataset(test_ds, "test", test_paths_raw)

    # Save a master CSV containing top-level final metrics
    with open(METRICS_PATH, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerow(["train_accuracy", f"{train_metrics['accuracy']:.4f}"])
        writer.writerow(["validation_accuracy", f"{val_metrics['accuracy']:.4f}"])
        writer.writerow(["test_accuracy", f"{test_metrics['accuracy']:.4f}"])
        writer.writerow(["test_precision_weighted", f"{test_metrics['precision']:.4f}"])
        writer.writerow(["test_recall_weighted", f"{test_metrics['recall']:.4f}"])
        writer.writerow(["test_f1_weighted", f"{test_metrics['f1']:.4f}"])
        for index, class_name in enumerate(class_names):
            writer.writerow([
                f"test_f1_{class_name.replace(' ', '_')}",
                f"{test_metrics['f1_per_class'][index]:.4f}"
            ])

    # Save a clean JSON summary of the overall run
    summary_json_path = os.path.join(
        BASE_DIR, "round11_summary_ResNet50.json"
    )
    with open(summary_json_path, "w") as file:
        json.dump({
            "round": "Round_11",
            "model": MODEL_NAME,
            "split": "70/20/10",
            "train_accuracy": train_metrics["accuracy"],
            "validation_accuracy": val_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"]
        }, file, indent=4)

    # Push all collected end-of-run metrics to MLflow
    mlflow.log_metrics({
        "train_accuracy": train_metrics["accuracy"],
        "train_precision_weighted": train_metrics["precision"],
        "train_recall_weighted": train_metrics["recall"],
        "train_f1_weighted": train_metrics["f1"],
        "validation_accuracy": val_metrics["accuracy"],
        "validation_precision_weighted": val_metrics["precision"],
        "validation_recall_weighted": val_metrics["recall"],
        "validation_f1_weighted": val_metrics["f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision_weighted": test_metrics["precision"],
        "test_recall_weighted": test_metrics["recall"],
        "test_f1_weighted": test_metrics["f1"],
        "best_validation_accuracy": max(history.history["val_accuracy"]),
        "best_epoch": int(
            np.argmax(history.history["val_accuracy"]) + 1
        )
    })

    # Prepare list of files to upload to MLflow's artifact storage
    artifact_files = [
        (__file__, "source"),                                # Upload this script itself
        (SPLIT_MANIFEST_PATH, "dataset_split"),              # Upload split configuration
        (MODEL_SAVE_PATH, "model"),                          # Upload compiled model
        (MODEL_SUMMARY_PATH, "model"),                       # Upload text model summary
        (HISTORY_PATH, "training"),                          # Upload epoch training history
        (METRICS_PATH, "results"),                           # Upload top-level metrics
        (summary_json_path, "results"),                      # Upload json summary
        (train_metrics["report_path"], "evaluation"),        # Upload Train evaluation docs
        (train_metrics["cm_path"], "evaluation"),
        (train_metrics["prediction_path"], "evaluation"),
        (val_metrics["report_path"], "evaluation"),          # Upload Validation evaluation docs
        (val_metrics["cm_path"], "evaluation"),
        (val_metrics["prediction_path"], "evaluation"),
        (test_metrics["report_path"], "evaluation"),         # Upload Test evaluation docs
        (test_metrics["cm_path"], "evaluation"),
        (test_metrics["prediction_path"], "evaluation")
    ]

    # Upload all available artifacts iteratively
    for file_path, artifact_folder in artifact_files:
        if os.path.exists(file_path):
            mlflow.log_artifact(file_path, artifact_path=artifact_folder)

    # Final read-out
    print("\n" + "=" * 70)
    print("Round 11 ResNet has been completed")
    print("=" * 70)
    print(f"Train Accuracy      : {train_metrics['accuracy'] * 100:.2f}%")
    print(f"Validation Accuracy : {val_metrics['accuracy'] * 100:.2f}%")
    print(f"Test Accuracy       : {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Test Precision      : {test_metrics['precision'] * 100:.2f}%")
    print(f"Test Recall         : {test_metrics['recall'] * 100:.2f}%")
    print(f"Test F1 Score       : {test_metrics['f1'] * 100:.2f}%")
    print("MLflow experiment:")
    print("PlastiSort_Round_11_70_20_10")
    print("=" * 70)
