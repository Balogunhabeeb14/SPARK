# ==============================================================================
# PLASTISORT AI: EFFICIENTNETB3 TRAINING & EVALUATION SCRIPT
# ==============================================================================
# Simple summary of what this code does:
# 1. Finds and organizes plastic waste image files into classes.
# 2. Splits images into 70% Training, 20% Validation, and 10% Testing sets.
# 3. Prepares an EfficientNetB3 deep learning model with data augmentation.
# 4. Trains the model while logging results with MLflow.
# 5. Evaluates accuracy, precision, recall, and F1-score across all data splits.
# 6. Saves the trained model, performance summaries, and evaluation reports.

# --- STEP 1: IMPORT REQUIRED LIBRARIES ---

# Operating system tools for creating folders, joining file paths, and reading directories
import os

# Tool for writing data into CSV spreadsheet files
import csv

# Tool for saving configuration data in standard JSON format
import json

# Tool for creating random numbers (used for reproducible dataset shuffling)
import random

# Advanced path-handling utility for MLflow URIs
from pathlib import Path

# Math library for handling numerical arrays and calculations
import numpy as np

# Deep learning framework used to build, train, and run neural networks
import tensorflow as tf

# Experiment tracking tool used to record metrics, parameters, and saved models
import mlflow

# Specific neural network building blocks from Keras (layers and model containers)
from tensorflow.keras import layers, models

# Machine learning evaluation tools from Scikit-Learn to measure performance
from sklearn.metrics import (
    classification_report,  # Generates text summary of precision, recall, and F1-score
    confusion_matrix,       # Creates table comparing true labels vs predicted labels
    accuracy_score,         # Calculates percentage of correct predictions
    precision_score,        # Measures how accurate positive predictions were
    recall_score,           # Measures how many actual positives were correctly caught
    f1_score                # Balanced score combining precision and recall
)

# Tool for randomly splitting lists into training, validation, and testing sets
from sklearn.model_selection import train_test_split

# Tool for calculating weight balances to prevent the model from favoring larger classes
from sklearn.utils.class_weight import compute_class_weight


# --- STEP 2: DEFINE FILE PATHS & DIRECTORIES ---

# Find the directory where this current Python script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Find the main project folder (one level above the script folder)
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Path to the main dataset folder containing class subfolders of plastic images
DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "ai_plastic_waste_management_training_updated"
)

# Path to the CSV file that locks in the dataset split so all models use identical images
SPLIT_MANIFEST_PATH = os.path.join(
    PROJECT_ROOT,
    "round11_grouped_split_70_20_10.csv"
)

# Folder where MLflow will store experiment tracking logs and run history
MLFLOW_DIR = os.path.join(
    PROJECT_ROOT,
    "mlruns"
)

# Print detected folders to the terminal screen
print(f"Project Root detected as: {PROJECT_ROOT}")
print(f"Dataset Path targeted at: {DATA_PATH}")


# --- STEP 3: SET HYPERPARAMETERS & CONFIGURATIONS ---

# Model backbone architecture name
MODEL_NAME = "EfficientNetB3"

# Standard width and height to resize all input images (224x224 pixels)
IMG_SIZE = (224, 224)

# Number of images processed together in a single step
BATCH_SIZE = 64

# Seed number used across random processes so results stay consistent every run
SEED = 123

# Total training passes through the entire training dataset
EPOCHS = 40

# Speed at which the neural network updates its internal weights
LEARNING_RATE = 1e-4

# Number of top layers in the pre-trained model to unfreeze for fine-tuning
UNFREEZE_LAYERS = 60

# Data split ratios (70% training, 20% validation, 10% testing)
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.20
TEST_SPLIT = 0.10

# Set to True if you want to overwrite and recreate the split manifest CSV file
FORCE_RECREATE_SPLIT = False

# Exact folder names of the 5 plastic categories being classified
VALID_CLASSES = [
    "HDPE Plastic",
    "LDPE Plastic",
    "PET Plastic",
    "PP Plastic",
    "PS Plastic"
]

# Supported image file formats
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png"
)

# File path where the final trained neural network model will be saved
MODEL_SAVE_PATH = os.path.join(
    BASE_DIR,
    "plastic_model_EfficientNetB3_round11.keras"
)

# File path for saving training loss and accuracy history across epochs
HISTORY_PATH = os.path.join(
    BASE_DIR,
    "training_history_EfficientNetB3_round11.csv"
)

# File path for saving overall model evaluation scores
METRICS_PATH = os.path.join(
    BASE_DIR,
    "evaluation_metrics_EfficientNetB3_round11.csv"
)

# File path for saving the visual layer structure description of the model
MODEL_SUMMARY_PATH = os.path.join(
    BASE_DIR,
    "model_summary_EfficientNetB3_round11.txt"
)


# --- STEP 4: ENSURE REPRODUCIBILITY & VERIFY FOLDERS ---

print("Setting random seeds for reproducibility")
def set_seed(seed):
    """Sets fixed seeds for random number generators so tests are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Lock in the random seed
set_seed(SEED)

print("Verifying directory structures")
# Check if the main dataset directory actually exists on the computer
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

# Check if every individual plastic class folder exists inside the dataset folder
for class_name in VALID_CLASSES:
    class_path = os.path.join(DATA_PATH, class_name)
    if not os.path.isdir(class_path):
        raise FileNotFoundError(
            f"Missing class folder:\n{class_path}"
        )
print("Directory checks passed successfully")


# --- STEP 5: INITIALIZE LOCAL MLFLOW TRACKING ---

print("Configuring Local MLflow Tracking Instance", flush=True)
# Ensure the MLflow log directory exists
os.makedirs(MLFLOW_DIR, exist_ok=True)

# Convert local folder path into a valid file URI format for MLflow
tracking_uri = Path(MLFLOW_DIR).resolve().as_uri()
mlflow.set_tracking_uri(tracking_uri)
print(f"MLflow local tracking URI set to: {tracking_uri}")

# Set the active experiment name under which all training runs are logged
mlflow.set_experiment("PlastiSort_Round_11_70_20_10")
print("MLflow experiment initialized.")


# --- STEP 6: DISCOVER IMAGES & MANAGE DATASET SPLIT ---

print("Scanning dataset directory for images", flush=True)
def load_dataset(root_dir):
    """
    Scans dataset folders, identifies image files, and groups related image variations
    (like grayscale duplicates) under a shared base filename key.
    """
    # Alphabetically sort class names to ensure consistent label numbers (0, 1, 2, 3, 4)
    class_names = sorted([
        folder
        for folder in os.listdir(root_dir)
        if folder in VALID_CLASSES
    ])

    image_dict = {}

    # Loop through each plastic class folder
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(root_dir, class_name)

        # Walk through all files and subfolders
        for root, _, files in os.walk(class_path):
            for file_name in files:
                # Only process supported image file types
                if file_name.lower().endswith(IMAGE_EXTENSIONS):
                    # Clean filename to group matching images (e.g. sample_1 and sample_1_gray)
                    base_name = (
                        file_name
                        .replace("_gray", "")
                        .replace("_grey", "")
                        .split(".")[0]
                    )

                    full_path = os.path.join(root, file_name)

                    # Initialize group entry if it hasn't been seen yet
                    if base_name not in image_dict:
                        image_dict[base_name] = {
                            "paths": [],
                            "label": label
                        }

                    # Add file path to this base group
                    image_dict[base_name]["paths"].append(full_path)

    return image_dict, class_names


# Run the scanning function to locate all available images
image_dict, class_names = load_dataset(DATA_PATH)
print(f"Scanning is complete complete. Parsed {len(image_dict)} base image groupings.")


def create_and_save_split():
    """
    Splits image groups into 70% Train, 20% Validation, and 10% Test sets
    while preserving class proportions (stratification), then writes to CSV.
    """
    print("Split manifest missing or forced. Creating a new train/val/test allocation...")
    all_groups = list(image_dict.keys())

    all_labels = [
        image_dict[group]["label"]
        for group in all_groups
    ]

    # Split 1: Extract 70% for Training, leaving 30% temporary pool for Validation + Test
    train_groups, temporary_groups = train_test_split(
        all_groups,
        test_size=(VAL_SPLIT + TEST_SPLIT),
        stratify=all_labels,
        random_state=SEED
    )

    temporary_labels = [
        image_dict[group]["label"]
        for group in temporary_groups
    ]

    # Split 2: Divide the 30% pool into Validation (20% total) and Test (10% total)
    validation_groups, test_groups = train_test_split(
        temporary_groups,
        test_size=(
            TEST_SPLIT /
            (VAL_SPLIT + TEST_SPLIT)
        ),
        stratify=temporary_labels,
        random_state=SEED
    )

    # Open CSV manifest file and write down split assignments for every image
    with open(
        SPLIT_MANIFEST_PATH,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)
        writer.writerow(["split", "path", "label"])

        # Write training set rows
        for group in train_groups:
            for path in image_dict[group]["paths"]:
                writer.writerow([
                    "train",
                    path,
                    image_dict[group]["label"]
                ])

        # Write validation set rows
        for group in validation_groups:
            for path in image_dict[group]["paths"]:
                writer.writerow([
                    "validation",
                    path,
                    image_dict[group]["label"]
                ])

        # Write test set rows
        for group in test_groups:
            for path in image_dict[group]["paths"]:
                writer.writerow([
                    "test",
                    path,
                    image_dict[group]["label"]
                ])

    print("Created Round 11 shared split file:")
    print(SPLIT_MANIFEST_PATH)


def load_split_from_csv():
    """Reads image file paths and numeric labels from the split CSV manifest."""
    print("📖 Reading split configurations from shared CSV manifest...")
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    # Read manifest line by line
    with open(
        SPLIT_MANIFEST_PATH,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            split_name = row["split"]
            path = row["path"]
            label = int(row["label"])

            # Verify that every file listed in the manifest exists on disk
            if not os.path.exists(path):
                raise FileNotFoundError(
                    "\nAn image listed in the split CSV "
                    "cannot be found:\n"
                    f"{path}"
                )

            # Route paths to their assigned split lists
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


# Create split manifest if it doesn't exist yet or if recreation is forced
if (
    not os.path.exists(SPLIT_MANIFEST_PATH)
    or FORCE_RECREATE_SPLIT
):
    create_and_save_split()

# Load image path lists for all three dataset splits
(
    train_paths_raw,
    train_labels_raw,
    val_paths_raw,
    val_labels_raw,
    test_paths_raw,
    test_labels_raw
) = load_split_from_csv()

# Calculate balanced class weights so rarer classes get higher importance during training
print("Calculating the balanced class weights")
class_weight_values = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels_raw),
    y=train_labels_raw
)

# Convert class weight values to a dictionary matching label indices (0, 1, 2, 3, 4)
class_weight_dict = dict(enumerate(class_weight_values))

print("Class weights (balanced):")
for index, class_name in enumerate(class_names):
    print(
        f"  {class_name:<20}: "
        f"{class_weight_dict[index]:.4f}"
    )

# --- STEP 7: PRINT DATASET SUMMARY TABLES ---

print("\n" + "=" * 70)
print("Round 11: EfficientNetB3")
print("Dataset Summary")
print("=" * 70)

print(f"Dataset path: {DATA_PATH}")
print(f"Plastic classes: {class_names}")
print(f"Total image groups: {len(image_dict)}")

# Count overall image totals for each plastic class
class_counts = {
    index: 0
    for index in range(len(class_names))
}

for group in image_dict.values():
    class_counts[group["label"]] += len(group["paths"])

print("Images per class:")
for index, class_name in enumerate(class_names):
    print(f"{class_name:<20}: {class_counts[index]}")

print("Grouped stratified split:")
print(f"Training images   : {len(train_paths_raw)}")
print(f"Validation images : {len(val_paths_raw)}")
print(f"Test images       : {len(test_paths_raw)}")

print("\nClass distribution per split:")
print(
    f"{'Class':<20} "
    f"{'Train':>8} "
    f"{'Val':>8} "
    f"{'Test':>8}"
)
print("-" * 55)

# Print specific class counts for each dataset split side by side
for index, class_name in enumerate(class_names):
    train_count = train_labels_raw.count(index)
    val_count = val_labels_raw.count(index)
    test_count = test_labels_raw.count(index)
    print(
        f"{class_name:<20} "
        f"{train_count:>8} "
        f"{val_count:>8} "
        f"{test_count:>8}"
    )

print("=" * 70)


# --- STEP 8: BUILD TENSORFLOW DATA PIPELINES ---

print("Constructing optimized tf.data input streaming pipelines", flush=True)
def process_image(path, label):
    """
    Reads an image file from disk, decodes it into RGB channels,
    resizes it to 224x224 pixels, and converts pixels to float32 values.
    EfficientNetB3 handles pixel normalization internally within its architecture.
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
    """
    Converts list of file paths and labels into a high-performance tf.data pipeline
    with parallel reading, optional shuffling, batching, and GPU prefetching.
    """
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )
    # Process images in parallel using available CPU cores
    dataset = dataset.map(
        process_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    # Shuffle order if requested (only during model training)
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=1000,
            seed=SEED,
            reshuffle_each_iteration=True
        )
    # Group images into mini-batches
    dataset = dataset.batch(BATCH_SIZE)
    # Prefetch next batch in memory so GPU never waits for disk reading
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# Create streaming datasets for training, evaluation, validation, and testing
train_ds = create_dataset(
    train_paths_raw, train_labels_raw, shuffle=True
)
train_eval_ds = create_dataset(
    train_paths_raw, train_labels_raw, shuffle=False
)
val_ds = create_dataset(
    val_paths_raw, val_labels_raw, shuffle=False
)
test_ds = create_dataset(
    test_paths_raw, test_labels_raw, shuffle=False
)
print("Data streaming pipelines successfully optimized with AUTOTUNE prefetching.")


# --- STEP 9: BUILD MODEL ARCHITECTURE & CUSTOM CALLBACKS ---

# Custom callback to automatically log training and validation metrics per epoch to MLflow
class MLflowMetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        metrics = {}
        for key, value in logs.items():
            if value is not None:
                metrics[f"epoch_{key}"] = float(value)
        # Record loss and accuracy values into MLflow for this epoch step
        mlflow.log_metrics(metrics, step=epoch + 1)


print("Building the EfficientNetB3 architecture and unfreezing fine-tuning layers", flush=True)

# Data augmentation block to randomly distort images during training to prevent overfitting
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),       # Random left-right flipping
    layers.RandomRotation(0.3),            # Random slight rotations
    layers.RandomZoom(0.2),                # Random zooming in/out
    layers.RandomTranslation(0.1, 0.1),    # Random slight image shifts
    layers.RandomBrightness(0.25),         # Random lighting adjustments
    layers.RandomContrast(0.25),           # Random contrast adjustments
    layers.GaussianNoise(0.03),            # Random subtle pixel noise
], name="data_augmentation")

# Load pre-trained EfficientNetB3 base model initialized with ImageNet visual weights
base_model = tf.keras.applications.EfficientNetB3(
    input_shape=(224, 224, 3),
    include_top=False,  # Exclude original 1000-class classification top layer
    weights="imagenet"
)

# Enable layer training on base model
base_model.trainable = True

# Freeze all lower feature-extraction layers except the top UNFREEZE_LAYERS (60) for fine-tuning
for layer in base_model.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False

# Stack final network architecture
model = models.Sequential([
    data_augmentation,                   # Apply image variations
    base_model,                          # Pre-trained feature extractor
    layers.GlobalAveragePooling2D(),     # Compress 2D feature maps to 1D vector
    layers.Dropout(0.5),                 # Randomly drop 50% nodes to stop overfitting
    layers.Dense(                        # Output layer with 5 probabilities using Softmax
        len(class_names),
        activation="softmax"
    )
], name="round11_EfficientNetB3")

# Compile model specifying optimization algorithm, loss function, and tracking metric
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Build network shapes explicitly using sample input dimensions
model.build(input_shape=(None, IMG_SIZE[0], IMG_SIZE[1], 3))

# Save text representation of full model architecture layers to disk
with open(MODEL_SUMMARY_PATH, "w") as file:
    model.summary(
        print_fn=lambda line: file.write(line + "\n")
    )

print("\n" + "=" * 70)
print("Model Architecture EfficientNetB3")
print("=" * 70)
model.summary()

# Configure Early Stopping to automatically stop training if validation accuracy stops improving
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=10,             # Wait 10 epochs without improvement before stopping
    min_delta=0.005,         # Minimum required improvement threshold
    restore_best_weights=True,  # Revert model to best performing epoch weights
    verbose=1
)


# --- STEP 10: DEFINE MODEL EVALUATION FUNCTION ---

def evaluate_dataset(dataset, split_name, paths):
    """
    Evaluates trained model predictions against real labels for a given split (train/val/test).
    Saves classification reports, confusion matrices, and detailed prediction outputs to disk.
    """
    print(f"Evaluating {split_name.upper()} performance profile...")
    
    # Extract true ground-truth labels from dataset generator batches
    true_labels = []
    for _, label_batch in dataset:
        true_labels.extend(label_batch.numpy())
    true_labels = np.array(true_labels)

    # Predict class probabilities across all images in dataset (verbose=1 shows progress bar)
    probabilities = model.predict(dataset, verbose=1)
    # Identify predicted class index with highest probability score
    predicted_classes = np.argmax(probabilities, axis=1)
    # Extract numerical confidence value (highest probability)
    confidence_scores = np.max(probabilities, axis=1)

    # Compute overall accuracy percentage
    acc = accuracy_score(true_labels, predicted_classes)
    
    # Compute weighted average evaluation scores
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
    
    # Compute per-class F1-scores individually for each plastic category
    f1_per_class = f1_score(
        true_labels, predicted_classes,
        labels=list(range(len(class_names))),
        average=None, zero_division=0
    )
    
    # Create full textual classification report breakdown
    report = classification_report(
        true_labels, predicted_classes,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4, zero_division=0
    )
    
    # Create numeric confusion matrix grid
    cm = confusion_matrix(
        true_labels, predicted_classes,
        labels=list(range(len(class_names)))
    )

    # Set destination file paths for evaluation exports
    report_path = os.path.join(
        BASE_DIR,
        f"classification_report_EfficientNetB3_round11_{split_name}.txt"
    )
    cm_path = os.path.join(
        BASE_DIR,
        f"confusion_matrix_EfficientNetB3_round11_{split_name}.csv"
    )
    prediction_path = os.path.join(
        BASE_DIR,
        f"prediction_results_EfficientNetB3_round11_{split_name}.csv"
    )

    # Write text report to file
    with open(report_path, "w") as file:
        file.write(report)

    # Write confusion matrix grid into CSV file format
    with open(cm_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["True / Predicted"] + class_names)
        for index, row in enumerate(cm):
            writer.writerow([class_names[index]] + row.tolist())

    # Write detailed image-by-image prediction output table into CSV
    with open(prediction_path, "w", newline="") as file:
        writer = csv.writer(file)
        header = [
            "image_path", "true_label", "true_class",
            "predicted_label", "predicted_class", "confidence"
        ]
        # Add probability headers for every plastic class column
        for class_name in class_names:
            header.append(
                f"probability_{class_name.replace(' ', '_')}"
            )
        writer.writerow(header)

        # Write row entry for each image evaluated
        for index in range(len(paths)):
            row = [
                paths[index],
                int(true_labels[index]),
                class_names[true_labels[index]],
                int(predicted_classes[index]),
                class_names[predicted_classes[index]],
                float(confidence_scores[index])
            ]
            # Append class probability distribution
            row.extend(probabilities[index].tolist())
            writer.writerow(row)

    # Display printed evaluation metrics to console
    print("\n" + "=" * 70)
    print(f"{split_name.upper()} Evaluation {MODEL_NAME}")
    print("=" * 70)
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "f1_per_class": f1_per_class,
        "report_path": report_path,
        "cm_path": cm_path,
        "prediction_path": prediction_path
    }


# --- STEP 11: EXECUTE TRAINING LOOP & LOG TO MLFLOW ---

print("Establishing active MLflow session run", flush=True)
# Start tracking run under MLflow
with mlflow.start_run(
    run_name="Round_11_EfficientNetB3_70_20_10"
):
    print("Registering system metadata tags and hyperparameters with the MLflow dashboard")
    # Log informational metadata tags
    mlflow.set_tags({
        "project": "PlastiSort AI",
        "round": "Round 11",
        "model_name": MODEL_NAME,
        "split": "70/20/10",
        "split_type": "grouped_stratified"
    })

    # Log training parameters and hyperparameters
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
    print("Starting the EfficientNetB3 Training Loop")
    print("=" * 70, flush=True)

    # Fit neural network model on training dataset while validating performance on validation set
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=[
            early_stopping,
            MLflowMetricsCallback()
        ],
        verbose=1
    )

    print("Training loop complete")
    # Save complete trained model file (.keras)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to:\n{MODEL_SAVE_PATH}")

    # Export epoch-by-epoch loss and accuracy metrics to CSV file
    with open(HISTORY_PATH, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch", "train_loss", "train_acc",
            "val_loss", "val_acc"
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

    # Evaluate final model against Training, Validation, and Testing sets
    print("\n🏁 Executing multi-split test validations...")
    train_metrics = evaluate_dataset(
        train_eval_ds, "train", train_paths_raw
    )
    val_metrics = evaluate_dataset(
        val_ds, "validation", val_paths_raw
    )
    test_metrics = evaluate_dataset(
        test_ds, "test", test_paths_raw
    )

    # Save metric summaries to simple CSV file
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

    # Save summary results into JSON format
    summary_json_path = os.path.join(
        BASE_DIR,
        "round11_summary_EfficientNetB3.json"
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

    # Push all final evaluation metrics directly into MLflow database entry
    print("Transferring the final metric evaluations to local MLflow database...")
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
        "best_validation_accuracy": max(
            history.history["val_accuracy"]
        ),
        "best_epoch": int(
            np.argmax(history.history["val_accuracy"]) + 1
        )
    })

    # List of generated files and model binaries to archive into MLflow artifact storage
    print("Uploading text logs, CSV manifests, and model binaries to MLflow Artifact repository")
    artifact_files = [
        (__file__, "source"),
        (SPLIT_MANIFEST_PATH, "dataset_split"),
        (MODEL_SAVE_PATH, "model"),
        (MODEL_SUMMARY_PATH, "model"),
        (HISTORY_PATH, "training"),
        (METRICS_PATH, "results"),
        (summary_json_path, "results"),
        (train_metrics["report_path"], "evaluation"),
        (train_metrics["cm_path"], "evaluation"),
        (train_metrics["prediction_path"], "evaluation"),
        (val_metrics["report_path"], "evaluation"),
        (val_metrics["cm_path"], "evaluation"),
        (val_metrics["prediction_path"], "evaluation"),
        (test_metrics["report_path"], "evaluation"),
        (test_metrics["cm_path"], "evaluation"),
        (test_metrics["prediction_path"], "evaluation")
    ]

    # Save output artifacts into MLflow run directory
    for file_path, artifact_folder in artifact_files:
        if os.path.exists(file_path):
            mlflow.log_artifact(
                file_path,
                artifact_path=artifact_folder
            )

    # Print final execution completion message and accuracy percentages to screen
    print("\n" + "=" * 70)
    print("Round 11 EfficientNetB3 execution finalized successfully!")
    print("=" * 70)
    print(f"Final Train Accuracy      : {train_metrics['accuracy'] * 100:.2f}%")
    print(f"Final Validation Accuracy : {val_metrics['accuracy'] * 100:.2f}%")
    print(f"Final Test Accuracy       : {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Final Test Precision      : {test_metrics['precision'] * 100:.2f}%")
    print(f"Final Test Recall         : {test_metrics['recall'] * 100:.2f}%")
    print(f"Final Test F1 Score       : {test_metrics['f1'] * 100:.2f}%")
    print("All run details have been pushed to local MLflow database workspace.")
    print("=" * 70)
