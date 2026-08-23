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

# PATHS: Define directory paths for the project, dataset, and output files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "ai_plastic_waste_management_training_updated"
)

# Shared split file for Round 3
# EfficientNetB3 must use the same file.
SPLIT_MANIFEST_PATH = os.path.join(
    PROJECT_ROOT,
    "round11_grouped_split_70_20_10.csv"
)

MLFLOW_DIR = os.path.join(
    PROJECT_ROOT,
    "mlruns"
)

# Configurations: Hyperparameters and training settings
MODEL_NAME = "ResNet50"

IMG_SIZE = (224, 224)
BATCH_SIZE = 64
SEED = 123
EPOCHS = 40
LEARNING_RATE = 1e-4
UNFREEZE_LAYERS = 60

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.20
TEST_SPLIT = 0.10

FORCE_RECREATE_SPLIT = False

VALID_CLASSES = [
    "HDPE Plastic",
    "LDPE Plastic",
    "PET Plastic",
    "PP Plastic",
    "PS Plastic"
]

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png"
)

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

# REPRODUCIBILITY: Set random seeds for consistent and reproducible results
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed(SEED)

# CHECK DATASET PATH: Ensure the main dataset directory and all class folders exist
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

# MLFLOW SETUP: Initialize MLflow tracking directory and experiment name
os.makedirs(MLFLOW_DIR, exist_ok=True)

mlflow.set_tracking_uri(
    Path(MLFLOW_DIR).resolve().as_uri()
)

mlflow.set_experiment(
    "PlastiSort_Round_11_70_20_10"
)

# LOAD DATASET: Scan directories and map image files to their respective labels
def load_dataset(root_dir):
    """
    Same method as Round 10.
    """
    class_names = sorted([
        folder
        for folder in os.listdir(root_dir)
        if folder in VALID_CLASSES
    ])

    image_dict = {}

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(root_dir, class_name)

        for root, _, files in os.walk(class_path):
            for file_name in files:
                if file_name.lower().endswith(IMAGE_EXTENSIONS):
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
                    image_dict[base_name]["paths"].append(full_path)

    return image_dict, class_names


image_dict, class_names = load_dataset(DATA_PATH)

# CREATE OR LOAD THE SHARED 70/20/10 SPLIT: Functions to split data or load an existing CSV manifest
def create_and_save_split():
    all_groups = list(image_dict.keys())
    all_labels = [
        image_dict[group]["label"] for group in all_groups
    ]

    train_groups, temporary_groups = train_test_split(
        all_groups,
        test_size=(VAL_SPLIT + TEST_SPLIT),
        stratify=all_labels,
        random_state=SEED
    )

    temporary_labels = [
        image_dict[group]["label"] for group in temporary_groups
    ]

    validation_groups, test_groups = train_test_split(
        temporary_groups,
        test_size=(TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT)),
        stratify=temporary_labels,
        random_state=SEED
    )

    with open(SPLIT_MANIFEST_PATH, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "path", "label"])

        for group in train_groups:
            for path in image_dict[group]["paths"]:
                writer.writerow(["train", path, image_dict[group]["label"]])

        for group in validation_groups:
            for path in image_dict[group]["paths"]:
                writer.writerow(["validation", path, image_dict[group]["label"]])

        for group in test_groups:
            for path in image_dict[group]["paths"]:
                writer.writerow(["test", path, image_dict[group]["label"]])

    print("Created Round 11 shared split file:")
    print(SPLIT_MANIFEST_PATH)

def load_split_from_csv():
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    with open(SPLIT_MANIFEST_PATH, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            split_name = row["split"]
            path = row["path"]
            label = int(row["label"])

            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"\nImage not found:\n{path}"
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


if not os.path.exists(SPLIT_MANIFEST_PATH) or FORCE_RECREATE_SPLIT:
    create_and_save_split()

(
    train_paths_raw, train_labels_raw,
    val_paths_raw, val_labels_raw,
    test_paths_raw, test_labels_raw
) = load_split_from_csv()

# CLASS WEIGHTS: Compute weights to handle dataset class imbalances (e.g., PS and PP)
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

# DATASET SUMMARY: Print overall statistics and distribution of the dataset and splits
print("\n" + "=" * 70)
print("ROUND 11 - RESNET50")
print("Dataset Summary")
print("=" * 70)

print(f"Dataset path: {DATA_PATH}")
print(f"Plastic classes: {class_names}")
print(f"Total image groups: {len(image_dict)}")

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

# IMAGE PROCESSING: Functions to load, decode, resize, and structure TensorFlow datasets
def process_image(path, label):
    """
    Same as Round 10.
    ResNet50 preprocessing applied by ResNet50Preprocess layer.
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
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        process_image, num_parallel_calls=tf.data.AUTOTUNE
    )
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=1000, seed=SEED,
            reshuffle_each_iteration=True
        )
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = create_dataset(train_paths_raw, train_labels_raw, shuffle=True)
train_eval_ds = create_dataset(train_paths_raw, train_labels_raw, shuffle=False)
val_ds = create_dataset(val_paths_raw, val_labels_raw, shuffle=False)
test_ds = create_dataset(test_paths_raw, test_labels_raw, shuffle=False)

# MLFLOW CALLBACK: Custom callback to log epoch-wise metrics to MLflow automatically
class MLflowMetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        metrics = {
            f"epoch_{key}": float(value)
            for key, value in logs.items()
            if value is not None
        }
        mlflow.log_metrics(metrics, step=epoch + 1)

# MODEL SETUP: Define data augmentation, base ResNet50 architecture, and custom preprocess layer
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(0.25),
    layers.RandomContrast(0.25),
    layers.GaussianNoise(0.03),
], name="data_augmentation")

base_model = tf.keras.applications.ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = True

for layer in base_model.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False

# RESNET50 PREPROCESSING LAYER
@tf.keras.utils.register_keras_serializable(
    package="PlastiSort"
)
class ResNet50Preprocess(layers.Layer):
    """
    Applies official ResNet50 preprocessing after augmentation.
    ResNet50 does not normalise inputs internally.
    """
    def call(self, inputs):
        return tf.keras.applications.resnet50.preprocess_input(
            inputs
        )

model = models.Sequential([
    data_augmentation,
    ResNet50Preprocess(name="resnet50_preprocess"),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation="softmax")
], name="round11_ResNet50")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.build(input_shape=(None, IMG_SIZE[0], IMG_SIZE[1], 3))

# MODEL SUMMARY: Export and print the model's structural summary
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

# EARLY STOPPING: Configure early stopping to halt training if validation accuracy plateaus
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    min_delta=0.005,
    restore_best_weights=True,
    verbose=1
)

# EVALUATION FUNCTION: Compute metrics, confusion matrices, and classification reports for any dataset split
def evaluate_dataset(dataset, split_name, paths):
    true_labels = []
    for _, label_batch in dataset:
        true_labels.extend(label_batch.numpy())
    true_labels = np.array(true_labels)

    probabilities = model.predict(dataset, verbose=0)
    predicted_classes = np.argmax(probabilities, axis=1)
    confidence_scores = np.max(probabilities, axis=1)

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

    with open(report_path, "w") as file:
        file.write(report)

    with open(cm_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["True / Predicted"] + class_names)
        for index, row in enumerate(cm):
            writer.writerow([class_names[index]] + row.tolist())

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

    return {
        "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1,
        "f1_per_class": f1_per_class,
        "report_path": report_path,
        "cm_path": cm_path,
        "prediction_path": prediction_path
    }

# TRAINING + MLFLOW RUN: Execute model training, evaluate splits, log parameters/metrics, and save artifacts
with mlflow.start_run(run_name="Round_11_ResNet50_70_20_10"):

    mlflow.set_tags({
        "project": "PlastiSort AI",
        "round": "Round 11",
        "model_name": MODEL_NAME,
        "split": "70/20/10",
        "split_type": "grouped_stratified"
    })

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

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, MLflowMetricsCallback()],
        verbose=1
    )

    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to:\n{MODEL_SAVE_PATH}")

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

    train_metrics = evaluate_dataset(train_eval_ds, "train", train_paths_raw)
    val_metrics = evaluate_dataset(val_ds, "validation", val_paths_raw)
    test_metrics = evaluate_dataset(test_ds, "test", test_paths_raw)

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

    for file_path, artifact_folder in artifact_files:
        if os.path.exists(file_path):
            mlflow.log_artifact(file_path, artifact_path=artifact_folder)

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
