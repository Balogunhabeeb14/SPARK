# Round 13 - MobileNetV3Large with Hyperparameter Tuning
# Method: KerasTuner RandomSearch

import os
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
import mlflow
import keras_tuner as kt

from tensorflow.keras import layers, models

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# Stability settings for Mac

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
except Exception:
    pass


# Paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_PATH_OPTIONS = [
    os.path.join(
        PROJECT_ROOT,
        "ai_plastic_waste_management_training"
    )
]

DATA_PATH = next(
    (
        path
        for path in DATA_PATH_OPTIONS
        if os.path.isdir(path)
    ),
    DATA_PATH_OPTIONS[0],
)

SPLIT_MANIFEST_PATH = os.path.join(
    PROJECT_ROOT,
    "round13_6class_fixed_grouped_split_70_20_10.csv",
)

MLFLOW_DIR = os.path.join(
    PROJECT_ROOT,
    "mlruns",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "round13_fast_tune_outputs",
)

TUNER_DIR = os.path.join(
    BASE_DIR,
    "round13_fast_randomsearch_tuner",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print(f"Project Root : {PROJECT_ROOT}")
print(f"Dataset Path : {DATA_PATH}")
print(f"Output Dir   : {OUTPUT_DIR}")


# Configuration

MODEL_NAME = "MobileNetV3Large_Round13_FastTune"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 123

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.20
TEST_SPLIT = 0.10

FORCE_RECREATE_SPLIT = False

SEARCH_MAX_TRIALS = 4
SEARCH_EPOCHS = 4
FINAL_EPOCHS = 20

FINAL_PATIENCE = 5
SEARCH_PATIENCE = 2

PP_PS_WEIGHT_BOOST = 1.3

VALID_CLASSES = [
    "HDPE Plastic",
    "LDPE Plastic",
    "PET Plastic",
    "PP Plastic",
    "PS Plastic",
    "UNKNOWN",
]

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
)

MODEL_SAVE_PATH = os.path.join(
    OUTPUT_DIR,
    "plastic_model_MobileNetV3Large_round13_fast_tuned.keras",
)

CHECKPOINT_PATH = os.path.join(
    OUTPUT_DIR,
    "checkpoint_best_MobileNetV3Large_round13_fast_tuned.keras",
)

BEST_HP_PATH = os.path.join(
    OUTPUT_DIR,
    "best_hyperparameters_round13_fast_tuned.json",
)

METRICS_JSON_PATH = os.path.join(
    OUTPUT_DIR,
    "round13_fast_tuned_metrics.json",
)

METRICS_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "round13_fast_tuned_metrics.csv",
)

HISTORY_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "training_history_round13_fast_tuned.csv",
)


# Reproducibility

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed(SEED)


# MLflow setup

os.makedirs(
    MLFLOW_DIR,
    exist_ok=True
)

mlflow.set_tracking_uri(
    Path(MLFLOW_DIR).resolve().as_uri()
)

mlflow.set_experiment(
    "PlastiSort_Round_13_Fast_Tuning"
)


# Check dataset folders

if not os.path.isdir(DATA_PATH):
    raise FileNotFoundError(
        f"\nDataset folder not found:\n{DATA_PATH}"
    )

for class_name in VALID_CLASSES:
    class_path = os.path.join(
        DATA_PATH,
        class_name
    )

    if not os.path.isdir(class_path):
        raise FileNotFoundError(
            f"\nMissing class folder:\n{class_path}"
        )


# Load dataset

def load_dataset(root_dir):
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
                    base_name = (
                        os.path.splitext(file_name)[0]
                        .replace("_gray", "")
                        .replace("_grey", "")
                    )

                    # Keep matching colour and grey images in the same group.
                    relative_parent = os.path.relpath(
                        root,
                        class_path,
                    )

                    normalized_parent = (
                        relative_parent
                        .replace("_gray", "")
                        .replace("_grey", "")
                    )

                    group_key = (
                        f"{class_name}::"
                        f"{normalized_parent}::"
                        f"{base_name}"
                    )

                    full_path = os.path.join(
                        root,
                        file_name
                    )

                    if group_key not in image_dict:
                        image_dict[group_key] = {
                            "paths": [],
                            "label": label,
                        }

                    image_dict[
                        group_key
                    ]["paths"].append(
                        full_path
                    )

    return image_dict, class_names


image_dict, class_names = load_dataset(
    DATA_PATH
)


# Create or load grouped 70/20/10 split

def create_and_save_split():
    print(
        "Creating grouped 70/20/10 split..."
    )

    all_groups = list(
        image_dict.keys()
    )

    all_labels = [
        image_dict[group]["label"]
        for group in all_groups
    ]

    train_groups, temp_groups = train_test_split(
        all_groups,
        test_size=(VAL_SPLIT + TEST_SPLIT),
        stratify=all_labels,
        random_state=SEED,
    )

    temp_labels = [
        image_dict[group]["label"]
        for group in temp_groups
    ]

    val_groups, test_groups = train_test_split(
        temp_groups,
        test_size=(
            TEST_SPLIT /
            (VAL_SPLIT + TEST_SPLIT)
        ),
        stratify=temp_labels,
        random_state=SEED,
    )

    with open(
        SPLIT_MANIFEST_PATH,
        "w",
        newline=""
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "split",
            "path",
            "label",
        ])

        for group in train_groups:
            for path in image_dict[group]["paths"]:
                writer.writerow([
                    "train",
                    path,
                    image_dict[group]["label"],
                ])

        for group in val_groups:
            for path in image_dict[group]["paths"]:
                writer.writerow([
                    "validation",
                    path,
                    image_dict[group]["label"],
                ])

        for group in test_groups:
            for path in image_dict[group]["paths"]:
                writer.writerow([
                    "test",
                    path,
                    image_dict[group]["label"],
                ])

    print(
        f"Split saved to:\n{SPLIT_MANIFEST_PATH}"
    )


def load_split_from_csv():
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
                    "\nImage path in split CSV does not exist:\n"
                    f"{path}"
                )

            if split_name == "train":
                train_paths.append(path)
                train_labels.append(label)

            elif split_name in [
                "validation",
                "val",
                "valid",
            ]:
                val_paths.append(path)
                val_labels.append(label)

            elif split_name == "test":
                test_paths.append(path)
                test_labels.append(label)

    return (
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        test_paths,
        test_labels,
    )


if (
    FORCE_RECREATE_SPLIT
    or not os.path.exists(SPLIT_MANIFEST_PATH)
):
    create_and_save_split()

(
    train_paths_raw,
    train_labels_raw,
    val_paths_raw,
    val_labels_raw,
    test_paths_raw,
    test_labels_raw,
) = load_split_from_csv()


# Class weights

classes_array = np.unique(
    train_labels_raw
)

class_weight_values = compute_class_weight(
    class_weight="balanced",
    classes=classes_array,
    y=train_labels_raw,
)

class_weight_dict = {
    int(class_id): float(weight)
    for class_id, weight in zip(
        classes_array,
        class_weight_values
    )
}

pp_idx = class_names.index(
    "PP Plastic"
)

ps_idx = class_names.index(
    "PS Plastic"
)

class_weight_dict[
    pp_idx
] *= PP_PS_WEIGHT_BOOST

class_weight_dict[
    ps_idx
] *= PP_PS_WEIGHT_BOOST

print("\nClass weights:")

for index, class_name in enumerate(class_names):
    print(
        f"{class_name:<20}: "
        f"{class_weight_dict[index]:.4f}"
    )


# Dataset summary

print("\n" + "=" * 70)
print("Dataset Summary")
print("=" * 70)

print(
    f"Classes           : {', '.join(class_names)}"
)

print(
    f"Image groups      : {len(image_dict)}"
)

print(
    f"Train images      : {len(train_paths_raw)}"
)

print(
    f"Validation images : {len(val_paths_raw)}"
)

print(
    f"Test images       : {len(test_paths_raw)}"
)

print("=" * 70)


# Image processing

def process_image(path, label):
    image = tf.io.read_file(path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )

    image = tf.image.resize(
        image,
        IMG_SIZE,
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    image.set_shape([
        IMG_SIZE[0],
        IMG_SIZE[1],
        3,
    ])

    return image, label


def create_dataset(
    paths,
    labels,
    shuffle=False
):
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    dataset = dataset.map(
        process_image,
        num_parallel_calls=2,
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(
                500,
                len(paths)
            ),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(1)

    return dataset


train_ds = create_dataset(
    train_paths_raw,
    train_labels_raw,
    shuffle=True,
)

train_eval_ds = create_dataset(
    train_paths_raw,
    train_labels_raw,
    shuffle=False,
)

val_ds = create_dataset(
    val_paths_raw,
    val_labels_raw,
    shuffle=False,
)

test_ds = create_dataset(
    test_paths_raw,
    test_labels_raw,
    shuffle=False,
)


# Custom macro F1 metric

@tf.keras.utils.register_keras_serializable(
    package="PlastiSort"
)
class SparseMacroF1(tf.keras.metrics.Metric):
    def __init__(
        self,
        num_classes,
        name="macro_f1",
        **kwargs
    ):
        super().__init__(
            name=name,
            **kwargs
        )

        self.num_classes = num_classes

        self.cm = self.add_weight(
            name="cm",
            shape=(
                num_classes,
                num_classes
            ),
            initializer="zeros",
            dtype=tf.float32,
        )

    def update_state(
        self,
        y_true,
        y_pred,
        sample_weight=None
    ):
        y_true = tf.cast(
            tf.reshape(
                y_true,
                [-1]
            ),
            tf.int32,
        )

        y_pred = tf.argmax(
            y_pred,
            axis=-1,
            output_type=tf.int32,
        )

        batch_cm = tf.math.confusion_matrix(
            y_true,
            y_pred,
            num_classes=self.num_classes,
            dtype=tf.float32,
        )

        self.cm.assign_add(
            batch_cm
        )

    def result(self):
        true_positives = tf.linalg.diag_part(
            self.cm
        )

        false_positives = (
            tf.reduce_sum(
                self.cm,
                axis=0,
            )
            - true_positives
        )

        false_negatives = (
            tf.reduce_sum(
                self.cm,
                axis=1,
            )
            - true_positives
        )

        precision = true_positives / (
            true_positives
            + false_positives
            + 1e-7
        )

        recall = true_positives / (
            true_positives
            + false_negatives
            + 1e-7
        )

        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
                + 1e-7
            )
        )

        return tf.reduce_mean(f1)

    def reset_state(self):
        self.cm.assign(
            tf.zeros_like(self.cm)
        )

    def get_config(self):
        config = super().get_config()

        config.update({
            "num_classes": self.num_classes,
        })

        return config


# Model builder for RandomSearch

def build_model(hp):
    optimizer_name = hp.Choice(
        "optimizer",
        values=[
            "adam",
            "rmsprop",
        ],
    )

    learning_rate = hp.Choice(
        "learning_rate",
        values=[
            1e-3,
            1e-4,
            5e-5,
        ],
    )

    dropout_rate = hp.Choice(
        "dropout_rate",
        values=[
            0.2,
            0.3,
            0.5,
        ],
    )

    unfreeze_layers = hp.Choice(
        "unfreeze_layers",
        values=[
            0,
            20,
        ],
    )

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip(
            "horizontal"
        ),
        layers.RandomRotation(
            0.08
        ),
        layers.RandomZoom(
            0.08
        ),
        layers.RandomTranslation(
            0.05,
            0.05
        ),
        layers.RandomBrightness(
            0.08
        ),
        layers.RandomContrast(
            0.08
        ),
    ], name="light_augmentation")

    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=(
            IMG_SIZE[0],
            IMG_SIZE[1],
            3,
        ),
        include_top=False,
        weights="imagenet",
    )

    if unfreeze_layers == 0:
        base_model.trainable = False

    else:
        base_model.trainable = True

        for layer in base_model.layers[
            :-unfreeze_layers
        ]:
            layer.trainable = False

    model = models.Sequential([
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(
            dropout_rate
        ),
        layers.Dense(
            256,
            activation="relu",
        ),
        layers.Dropout(
            dropout_rate
        ),
        layers.Dense(
            len(class_names),
            activation="softmax",
            dtype="float32",
        ),
    ], name="MobileNetV3Large_fast_tuned")

    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        )

    else:
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate
        )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            SparseMacroF1(
                num_classes=len(class_names)
            ),
        ],
    )

    return model


# Evaluation helper

def evaluate_model(
    model,
    dataset,
    split_name
):
    true_labels = []
    predicted_labels = []

    for image_batch, label_batch in dataset:
        probs = model.predict(
            image_batch,
            verbose=0,
        )

        preds = np.argmax(
            probs,
            axis=1,
        )

        true_labels.extend(
            label_batch.numpy().tolist()
        )

        predicted_labels.extend(
            preds.tolist()
        )

    true_labels = np.array(
        true_labels
    )

    predicted_labels = np.array(
        predicted_labels
    )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    precision_weighted = precision_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    recall_weighted = recall_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    f1_weighted = f1_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    precision_macro = precision_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        labels=list(
            range(len(class_names))
        ),
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(
            range(len(class_names))
        ),
    )

    report_path = os.path.join(
        OUTPUT_DIR,
        f"classification_report_{split_name}.txt",
    )

    cm_path = os.path.join(
        OUTPUT_DIR,
        f"confusion_matrix_{split_name}.csv",
    )

    with open(
        report_path,
        "w"
    ) as file:
        file.write(report)

    with open(
        cm_path,
        "w",
        newline=""
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            ["True / Predicted"]
            + class_names
        )

        for index, row in enumerate(cm):
            writer.writerow(
                [class_names[index]]
                + row.tolist()
            )

    return {
        "accuracy": accuracy,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "report_path": report_path,
        "cm_path": cm_path,
    }


# RandomSearch tuning

print("\n" + "=" * 70)
print(
    "Starting FAST RandomSearch hyperparameter tuning"
)
print("=" * 70)

if os.path.isdir(TUNER_DIR):
    print(
        "Removing old fast tuner directory..."
    )

    shutil.rmtree(
        TUNER_DIR
    )


tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective=kt.Objective(
        "val_macro_f1",
        direction="max",
    ),
    max_trials=SEARCH_MAX_TRIALS,
    executions_per_trial=1,
    directory=TUNER_DIR,
    project_name="fast_randomsearch",
    overwrite=True,
    seed=SEED,
)

tuner.search_space_summary()


search_early_stopping = (
    tf.keras.callbacks.EarlyStopping(
        monitor="val_macro_f1",
        mode="max",
        patience=SEARCH_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
)


tuner.search(
    train_ds,
    validation_data=val_ds,
    epochs=SEARCH_EPOCHS,
    class_weight=class_weight_dict,
    callbacks=[
        search_early_stopping,
    ],
    verbose=1,
)


best_hp = tuner.get_best_hyperparameters(
    num_trials=1
)[0]

best_hyperparameters = {
    "optimizer":
        best_hp.get("optimizer"),

    "learning_rate":
        best_hp.get("learning_rate"),

    "dropout_rate":
        best_hp.get("dropout_rate"),

    "unfreeze_layers":
        best_hp.get("unfreeze_layers"),
}

with open(
    BEST_HP_PATH,
    "w"
) as file:
    json.dump(
        best_hyperparameters,
        file,
        indent=4,
    )

print("\nBest hyperparameters:")

for key, value in best_hyperparameters.items():
    print(
        f"{key}: {value}"
    )


# Final training

print("\n" + "=" * 70)
print(
    "Training final model using best hyperparameters"
)
print("=" * 70)

model = tuner.hypermodel.build(
    best_hp
)

final_early_stopping = (
    tf.keras.callbacks.EarlyStopping(
        monitor="val_macro_f1",
        mode="max",
        patience=FINAL_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    CHECKPOINT_PATH,
    monitor="val_macro_f1",
    mode="max",
    save_best_only=True,
    verbose=1,
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1,
)


with mlflow.start_run(
    run_name="Round_13_MobileNetV3Large_Fast_RandomSearch"
):
    mlflow.set_tags({
        "project": "PlastiSort AI",
        "round": "Round 13",
        "model": MODEL_NAME,
        "tuning_method": "RandomSearch",
        "objective": "val_macro_f1",
        "reason": "fast_stable_mac_training",
    })

    mlflow.log_params({
        "batch_size": BATCH_SIZE,
        "search_max_trials": SEARCH_MAX_TRIALS,
        "search_epochs": SEARCH_EPOCHS,
        "final_epochs": FINAL_EPOCHS,
        "pp_ps_weight_boost": PP_PS_WEIGHT_BOOST,
        **{
            f"best_{key}": value
            for key, value
            in best_hyperparameters.items()
        },
    })

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINAL_EPOCHS,
        class_weight=class_weight_dict,
        callbacks=[
            final_early_stopping,
            checkpoint,
            reduce_lr,
        ],
        verbose=1,
    )

    model.save(
        MODEL_SAVE_PATH
    )

    print(
        f"\nFinal model saved to:\n{MODEL_SAVE_PATH}"
    )


    # Save training history

    history_data = history.history

    with open(
        HISTORY_CSV_PATH,
        "w",
        newline=""
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "loss",
            "accuracy",
            "macro_f1",
            "val_loss",
            "val_accuracy",
            "val_macro_f1",
        ])

        for index in range(
            len(history_data["loss"])
        ):
            writer.writerow([
                index + 1,
                history_data["loss"][index],
                history_data["accuracy"][index],
                history_data["macro_f1"][index],
                history_data["val_loss"][index],
                history_data["val_accuracy"][index],
                history_data["val_macro_f1"][index],
            ])


    # Final evaluation

    train_metrics = evaluate_model(
        model,
        train_eval_ds,
        "train",
    )

    val_metrics = evaluate_model(
        model,
        val_ds,
        "validation",
    )

    test_metrics = evaluate_model(
        model,
        test_ds,
        "test",
    )


    # Save metrics

    all_metrics = {
        "model": MODEL_NAME,
        "tuning_method": "RandomSearch",
        "best_hyperparameters":
            best_hyperparameters,
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }

    clean_metrics = {
        "model": MODEL_NAME,
        "tuning_method": "RandomSearch",
        "best_hyperparameters":
            best_hyperparameters,

        "train": {
            key: value
            for key, value
            in train_metrics.items()
            if not key.endswith("_path")
        },

        "validation": {
            key: value
            for key, value
            in val_metrics.items()
            if not key.endswith("_path")
        },

        "test": {
            key: value
            for key, value
            in test_metrics.items()
            if not key.endswith("_path")
        },
    }

    with open(
        METRICS_JSON_PATH,
        "w"
    ) as file:
        json.dump(
            clean_metrics,
            file,
            indent=4,
        )

    with open(
        METRICS_CSV_PATH,
        "w",
        newline=""
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "split",
            "accuracy",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "precision_macro",
            "recall_macro",
            "f1_macro",
        ])

        for split_name, metrics in [
            ("train", train_metrics),
            ("validation", val_metrics),
            ("test", test_metrics),
        ]:
            writer.writerow([
                split_name,
                metrics["accuracy"],
                metrics["precision_weighted"],
                metrics["recall_weighted"],
                metrics["f1_weighted"],
                metrics["precision_macro"],
                metrics["recall_macro"],
                metrics["f1_macro"],
            ])

    mlflow.log_metrics({
        "train_accuracy":
            float(train_metrics["accuracy"]),

        "train_f1_macro":
            float(train_metrics["f1_macro"]),

        "validation_accuracy":
            float(val_metrics["accuracy"]),

        "validation_f1_macro":
            float(val_metrics["f1_macro"]),

        "test_accuracy":
            float(test_metrics["accuracy"]),

        "test_f1_macro":
            float(test_metrics["f1_macro"]),

        "best_val_macro_f1":
            float(
                max(
                    history.history[
                        "val_macro_f1"
                    ]
                )
            ),
    })

    artifact_files = [
        MODEL_SAVE_PATH,
        CHECKPOINT_PATH,
        BEST_HP_PATH,
        METRICS_JSON_PATH,
        METRICS_CSV_PATH,
        HISTORY_CSV_PATH,
        train_metrics["report_path"],
        train_metrics["cm_path"],
        val_metrics["report_path"],
        val_metrics["cm_path"],
        test_metrics["report_path"],
        test_metrics["cm_path"],
    ]

    for file_path in artifact_files:
        if os.path.exists(file_path):
            mlflow.log_artifact(
                file_path
            )


# Final output

print("\n" + "=" * 70)
print("FAST TUNING COMPLETE")
print("=" * 70)

print("Best hyperparameters:")

for key, value in best_hyperparameters.items():
    print(
        f"{key}: {value}"
    )

print("-" * 70)

print(
    f"Train Accuracy      : "
    f"{train_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"Validation Accuracy : "
    f"{val_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"Test Accuracy       : "
    f"{test_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"Test Weighted F1    : "
    f"{test_metrics['f1_weighted'] * 100:.2f}%"
)

print(
    f"Test Macro F1       : "
    f"{test_metrics['f1_macro'] * 100:.2f}%"
)

print("-" * 70)

print(
    f"Model saved at:\n{MODEL_SAVE_PATH}"
)

print(
    f"Best hyperparameters saved at:\n{BEST_HP_PATH}"
)

print(
    f"All outputs saved at:\n{OUTPUT_DIR}"
)

print("=" * 70)
