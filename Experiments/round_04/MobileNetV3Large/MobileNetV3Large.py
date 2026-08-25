# Purpose:
# 1. Builds a grouped 70/20/10 train/val/test dataset split.
# 2. Computes class weights with extra protection for PP/PS classes.
# 3. Fine-tunes MobileNetV3Large using TensorFlow/Keras.
# 4. Exports trained models (native Keras & FP16 TFLite for edge devices).
# 5. Evaluates model across train, validation, and test splits.
# 6. Logs parameters, metrics, and output artifacts to MLflow.

# Importing all the necessary libraries
import os
import csv
import json
import random
from pathlib import Path
import numpy as np
import tensorflow as tf
import mlflow
from tensorflow.keras import layers, models

# Import evaluation metrics from scikit-learn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Import utilities for dataset splitting and class imbalance handling
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# DIRECTORY PATH CONFIGURATION
# ============================================================

# Locate script base directory and project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Primary dataset folder path
DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "ai_plastic_waste_management_training_updated"
)

# Shared CSV manifest path to persist the 70/20/10 split across training runs
SPLIT_MANIFEST_PATH = os.path.join(
    PROJECT_ROOT,
    "round12_grouped_split_70_20_10.csv"
)

# Local MLflow tracking storage directory
MLFLOW_DIR = os.path.join(
    PROJECT_ROOT,
    "mlruns"
)

print(f"Project Root detected as: {PROJECT_ROOT}")
print(f"Dataset Path targeted at: {DATA_PATH}")

# ============================================================
# HYPERPARAMETERS & SYSTEM CONFIGURATION
# ============================================================

MODEL_NAME = "MobileNetV3Large_Round12"

# Image & batch processing configuration
IMG_SIZE = (224, 224)   # Input resolution expected by MobileNetV3
BATCH_SIZE = 64         # Number of samples per gradient update
SEED = 123              # Master random seed for reproducible results
EPOCHS = 40             # Maximum training epochs
LEARNING_RATE = 1e-4    # Initial learning rate for Adam optimizer
UNFREEZE_LAYERS = 60    # Number of top layers to unfreeze during fine-tuning

# Split ratio configuration (70% Train, 20% Validation, 10% Test)
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.20
TEST_SPLIT = 0.10

# Toggle to force regeneration of the split manifest file if set to True
FORCE_RECREATE_SPLIT = False

# Expected target plastic material classes
VALID_CLASSES = [
    "HDPE Plastic",
    "LDPE Plastic",
    "PET Plastic",
    "PP Plastic",
    "PS Plastic"
]

# Supported image file extensions
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png"
)

# Output paths for models, metrics, and training logs
MODEL_SAVE_PATH = os.path.join(
    BASE_DIR,
    "plastic_model_MobileNetV3Large_round12.keras"
)

TFLITE_SAVE_PATH = os.path.join(
    BASE_DIR,
    "mobilenetv3_plastic_fp16.tflite"
)

HISTORY_PATH = os.path.join(
    BASE_DIR,
    "training_history_MobileNetV3Large_round12.csv"
)

METRICS_PATH = os.path.join(
    BASE_DIR,
    "evaluation_metrics_MobileNetV3Large_round12.csv"
)

# ============================================================
# REPRODUCIBILITY & DIRECTORY VALIDATION
# ============================================================

print("🧬 Setting random seeds for reproducibility...")
def set_seed(seed):
    """Sets random seeds across Python, NumPy, and TensorFlow for determinism."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(SEED)

# Verify that the primary dataset directory exists
print("Verifying directory structures")
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

# Verify that subdirectories exist for each target plastic class
for class_name in VALID_CLASSES:
    class_path = os.path.join(DATA_PATH, class_name)
    if not os.path.isdir(class_path):
        raise FileNotFoundError(
            f"Missing class folder:\n{class_path}"
        )

# ============================================================
# MLFLOW SETUP
# ============================================================

os.makedirs(MLFLOW_DIR, exist_ok=True)

# Configure local file tracking URI and set experiment group
mlflow.set_tracking_uri(
    Path(MLFLOW_DIR).resolve().as_uri()
)

mlflow.set_experiment("PlastiSort_Round_12_70_20_10")

# ============================================================
# DATASET LOADING & GROUPING
# ============================================================

def load_dataset(root_dir):
    """
    Scans dataset folder and groups related image variants (e.g., color vs. grayscale)
    by base file name to prevent data leakage across splits.
    """
    class_names = sorted([
        folder
        for folder in os.listdir(root_dir)
        if folder in VALID_CLASSES
    ])

    image_dict = {}

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(
            root_dir,
            class_name
        )

        for root, _, files in os.walk(class_path):
            for file_name in files:
                if file_name.lower().endswith(
                    IMAGE_EXTENSIONS
                ):
                    # Remove suffix modifications to isolate unique physical object groups
                    base_name = (
                        file_name
                        .replace("_gray", "")
                        .replace("_grey", "")
                        .split(".")[0]
                    )

                    full_path = os.path.join(
                        root,
                        file_name
                    )

                    if base_name not in image_dict:
                        image_dict[base_name] = {
                            "paths": [],
                            "label": label
                        }

                    image_dict[base_name]["paths"].append(
                        full_path
                    )

    return image_dict, class_names


# Index raw dataset images into grouped image dictionary
image_dict, class_names = load_dataset(DATA_PATH)

# ============================================================
# DATASET SPLITTING (70/20/10 MANIFEST)
# ============================================================

def create_and_save_split():
    """Performs a stratified group split (70% train, 20% val, 10% test) and writes it to a CSV manifest."""
    print("⚠️ Split manifest missing or forced. Creating a new train/val/test allocation...")
    all_groups = list(image_dict.keys())

    all_labels = [
        image_dict[group]["label"]
        for group in all_groups
    ]

    # Step 1: Separate 70% training groups from 30% temporary (val + test) groups
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

    # Step 2: Split the 30% temporary set into 20% validation and 10% test groups
    validation_groups, test_groups = train_test_split(
        temporary_groups,
        test_size=(
            TEST_SPLIT /
            (VAL_SPLIT + TEST_SPLIT)
        ),
        stratify=temporary_labels,
        random_state=SEED
    )

    # Step 3: Write out explicit path-to-split mapping to disk CSV file
    with open(
        SPLIT_MANIFEST_PATH,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)
        writer.writerow(["split", "path", "label"])

        for group in train_groups:
            for path in image_dict[group]["paths"]:
                writer.writerow([
                    "train",
                    path,
                    image_dict[group]["label"]
                ])

        for group in validation_groups:
            for path in image_dict[group]["paths"]:
                writer.writerow([
                    "validation",
                    path,
                    image_dict[group]["label"]
                ])

        for group in test_groups:
            for path in image_dict[group]["paths"]:
                writer.writerow([
                    "test",
                    path,
                    image_dict[group]["label"]
                ])

    print("Created Round 12 shared split file:")
    print(SPLIT_MANIFEST_PATH)


def load_split_from_csv():
    """Reads paths and numerical ground-truth labels from the split CSV manifest."""
    train_paths = []
    train_labels = []
    val_paths = []
    val_labels = []
    test_paths = []
    test_labels = []

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

            if not os.path.exists(path):
                raise FileNotFoundError(
                    "\nAn image listed in the split CSV "
                    "cannot be found:\n"
                    f"{path}"
                )

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


# Generate manifest if it does not exist or if recreate is explicitly requested
if (
    not os.path.exists(SPLIT_MANIFEST_PATH)
    or FORCE_RECREATE_SPLIT
):
    create_and_save_split()

# Extract partitioned dataset lists from CSV manifest
(
    train_paths_raw,
    train_labels_raw,
    val_paths_raw,
    val_labels_raw,
    test_paths_raw,
    test_labels_raw
) = load_split_from_csv()

# ============================================================
# CLASS WEIGHT COMPUTATION & TARGETED BOOSTING
# ============================================================

# Compute standard balanced class weights to address dataset frequency imbalances
class_weight_values = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels_raw),
    y=train_labels_raw
)

class_weight_dict = dict(enumerate(class_weight_values))

# Apply targeted 30% weight multiplier to PP and PS classes to boost minority class performance
pp_idx = class_names.index("PP Plastic")
ps_idx = class_names.index("PS Plastic")
class_weight_dict[pp_idx] *= 1.3
class_weight_dict[ps_idx] *= 1.3

print("Class weights (balanced + PP/PS boost protection):")
for index, class_name in enumerate(class_names):
    print(
        f"  {class_name:<20}: "
        f"{class_weight_dict[index]:.4f}"
    )

# ============================================================
# DATASET SUMMARY & PRINT STATS
# ============================================================

print("\n" + "=" * 70)
print(f"Round 12: {MODEL_NAME}")
print("Dataset Summary")
print("=" * 70)

print(f"Dataset path: {DATA_PATH}")
print(f"Plastic classes: {class_names}")
print(f"Total image groups: {len(image_dict)}")

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

# ============================================================
# TENSORFLOW INPUT PIPELINE
# ============================================================

def process_image(path, label):
    """Loads image file from disk, decodes, resizes, and casts to float32 tensor."""
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    image.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return image, label


def create_dataset(paths, labels, shuffle=False):
    """Constructs an optimized tf.data dataset with mapping, optional shuffling, batching, and prefetching."""
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )
    dataset = dataset.map(
        process_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=1000,
            seed=SEED,
            reshuffle_each_iteration=True
        )
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# Prepare dataset iterators for training, validation, and testing
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

# ============================================================
# MLFLOW TRAINING CALLBACK
# ============================================================

class MLflowMetricsCallback(tf.keras.callbacks.Callback):
    """Custom Keras callback to log per-epoch training and validation metrics to MLflow."""
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        metrics = {}
        for key, value in logs.items():
            if value is not None:
                metrics[f"epoch_{key}"] = float(value)
        mlflow.log_metrics(metrics, step=epoch + 1)

# ============================================================
# MODEL ARCHITECTURE & DATA AUGMENTATION
# ============================================================

# Bin-realistic data augmentation pipeline tuned for waste sorting camera environments
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(0.15),
    layers.RandomContrast(0.15),
], name="data_augmentation")

# Load pre-trained MobileNetV3Large base architecture initialized with ImageNet weights
base_model = tf.keras.applications.MobileNetV3Large(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Enable fine-tuning on the top layer blocks while freezing earlier feature extraction layers
base_model.trainable = True

for layer in base_model.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False

# MobileNetV3 input scaling custom layer (registered for Keras model serialization)
@tf.keras.utils.register_keras_serializable(
    package="PlastiSort"
)
class MobileNetV3Preprocess(layers.Layer):
    """Applies official MobileNetV3 normalization preprocess function."""
    def call(self, inputs):
        return tf.keras.applications.mobilenet_v3.preprocess_input(
            inputs
        )

# Construct final Keras Sequential classification model
model = models.Sequential([
    data_augmentation,
    MobileNetV3Preprocess(),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(
        len(class_names),
        activation="softmax"
    )], name="round12_MobileNetV3Large")

# Compile model with Adam optimizer and sparse categorical loss
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Build model shape explicitly
model.build(input_shape=(None, IMG_SIZE[0], IMG_SIZE[1], 3))

# Save text summary of model architecture to disk
MODEL_SUMMARY_PATH = os.path.join(
    BASE_DIR,
    "model_summary_MobileNetV3Large_round12.txt"
)

with open(MODEL_SUMMARY_PATH, "w") as file:
    model.summary(
        print_fn=lambda line: file.write(line + "\n")
    )

print("\n" + "=" * 70)
print(f"Model Architecture {MODEL_NAME}")
print("=" * 70)
model.summary()

# Configure Early Stopping callback to halt training when validation accuracy plateaus
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    min_delta=0.005,
    restore_best_weights=True,
    verbose=1
)

# ============================================================
# EVALUATION ROUTINE
# ============================================================

def evaluate_dataset(dataset, split_name, paths):
    """
    Evaluates model predictions on a given dataset split, generates performance statistics,
    and writes out classification reports, confusion matrices, and detailed prediction CSVs.
    """
    true_labels = []
    predicted_classes = []
    confidence_scores = []
    probabilities = []

    # Iterate through batch predictions to keep path and prediction arrays strictly aligned
    for img_batch, label_batch in dataset:
        true_labels.extend(label_batch.numpy())
        preds = model(img_batch, training=False).numpy()
        probabilities.extend(preds)
        predicted_classes.extend(np.argmax(preds, axis=1))
        confidence_scores.extend(np.max(preds, axis=1))

    true_labels = np.array(true_labels)
    predicted_classes = np.array(predicted_classes)
    confidence_scores = np.array(confidence_scores)
    probabilities = np.array(probabilities)

    # Calculate overall aggregate metrics
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

    # Output filenames for generated artifacts
    report_path = os.path.join(
        BASE_DIR,
        f"classification_report_MobileNetV3Large_round12_{split_name}.txt"
    )
    cm_path = os.path.join(
        BASE_DIR,
        f"confusion_matrix_MobileNetV3Large_round12_{split_name}.csv"
    )
    prediction_path = os.path.join(
        BASE_DIR,
        f"prediction_results_MobileNetV3Large_round12_{split_name}.csv"
    )

    # Save classification text report
    with open(report_path, "w") as file:
        file.write(report)

    # Save confusion matrix CSV
    with open(cm_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["True / Predicted"] + class_names)
        for index, row in enumerate(cm):
            writer.writerow([class_names[index]] + row.tolist())

    # Save granular sample-level predictions CSV
    with open(prediction_path, "w", newline="") as file:
        writer = csv.writer(file)
        header = [
            "image_path", "true_label", "true_class",
            "predicted_label", "predicted_class", "confidence"
        ]
        for class_name in class_names:
            header.append(
                f"probability_{class_name.replace(' ', '_')}"
            )
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

# ============================================================
# MLFLOW TRACKED EXECUTION RUN
# ============================================================

with mlflow.start_run(run_name="Round_12_MobileNetV3Large_70_20_10"):

    # Log MLflow tags
    mlflow.set_tags({
        "project": "PlastiSort AI",
        "round": "Round 12",
        "model_name": MODEL_NAME,
        "split": "70/20/10",
        "split_type": "grouped_stratified"
    })

    # Log training hyperparameters to MLflow
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
            "RandomFlip(horizontal_and_vertical), "
            "RandomRotation(0.15), "
            "RandomZoom(0.15), "
            "RandomTranslation(0.1, 0.1), "
            "RandomBrightness(0.15), "
            "RandomContrast(0.15)"
        ),
        "class_weighting": "balanced_boosted",
        "train_images": len(train_paths_raw),
        "validation_images": len(val_paths_raw),
        "test_images": len(test_paths_raw)
    })

    print("\n" + "=" * 70)
    print("Beginning Training Run", flush=True)
    print("=" * 70, flush=True)

    # Train model using defined callbacks and class weights
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

    # Save full native Keras model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to:\n{MODEL_SAVE_PATH}")

    # Compile and convert model directly into optimized Float16 TFLite binary for Raspberry Pi / edge deployment
    print("Converting model to optimized Float16 TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with open(TFLITE_SAVE_PATH, "wb") as file:
        file.write(tflite_model)
    print(f"Optimized TFLite binary saved to:\n{TFLITE_SAVE_PATH}")

    # Export CSV containing epoch-by-epoch loss and accuracy metrics
    with open(HISTORY_PATH, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        
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

    # Evaluate across train, validation, and test datasets
    train_metrics = evaluate_dataset(train_eval_ds, "train", train_paths_raw)
    val_metrics = evaluate_dataset(val_ds, "validation", val_paths_raw)
    test_metrics = evaluate_dataset(test_ds, "test", test_paths_raw)

    # Export overall metrics summary file
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

    # Save JSON evaluation summary file
    summary_json_path = os.path.join(
        BASE_DIR,
        "round12_summary_MobileNetV3Large.json"
    )

    with open(summary_json_path, "w") as file:
        json.dump({
            "round": "Round_12",
            "model": MODEL_NAME,
            "split": "70/20/10",
            "train_accuracy": train_metrics["accuracy"],
            "validation_accuracy": val_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"]
        }, file, indent=4)

    # Log run metrics to MLflow
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

    # Define list of output files to register as MLflow artifacts
    artifact_files = [
        (__file__, "source"),
        (SPLIT_MANIFEST_PATH, "dataset_split"),
        (MODEL_SAVE_PATH, "model"),
        (TFLITE_SAVE_PATH, "model"),
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

    # Log existing artifacts to their designated MLflow target subfolders
    for file_path, artifact_folder in artifact_files:
        if os.path.exists(file_path):
            mlflow.log_artifact(
                file_path,
                artifact_path=artifact_folder
            )

    # Print final console execution summary
    print("\n" + "=" * 70)
    print("Round 12 MobileNetV3Large execution has been completed successfully")
    print("=" * 70)
    print(f"Final Train Accuracy      : {train_metrics['accuracy'] * 100:.2f}%")
    print(f"Final Validation Accuracy : {val_metrics['accuracy'] * 100:.2f}%")
    print(f"Final Test Accuracy       : {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Final Test Precision      : {test_metrics['precision'] * 100:.2f}%")
    print(f"Final Test Recall         : {test_metrics['recall'] * 100:.2f}%")
    print(f"Final Test F1 Score       : {test_metrics['f1'] * 100:.2f}%")
    print("All run details have been pushed to local MLflow database workspace.")
    print("=" * 70)
