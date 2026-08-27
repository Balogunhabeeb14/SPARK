# Round 9 - ResNet50 Prediction 

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# Configuration

MODEL_NAME = "ResNet50"

PREDICT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PREDICT_DIR.parent

IMG_SIZE = (224, 224)
BATCH_SIZE = 64

VALID_CLASSES = [
    "HDPE Plastic",
    "LDPE Plastic",
    "PET Plastic",
    "PP Plastic",
    "PS Plastic",
]

SPLIT_MANIFEST_PATH = (
    PROJECT_ROOT / "round9_grouped_split_70_20_10.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "Train"
    / "plastic_model_ResNet50_round9.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "round9_ResNet50_prediction_outputs"
)

MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT_NAME = "PlastiSort_Round_9_70_20_10"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MLFLOW_TRACKING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

mlflow.set_tracking_uri(
    f"file://{MLFLOW_TRACKING_DIR.resolve()}"
)

mlflow.set_experiment(
    MLFLOW_EXPERIMENT_NAME
)


# Custom layer required to load the ResNet50 model

@tf.keras.utils.register_keras_serializable(
    package="PlastiSort"
)
class ResNet50Preprocess(layers.Layer):
    def call(self, inputs):
        return tf.keras.applications.resnet50.preprocess_input(
            inputs
        )


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


def create_dataset(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    ds = ds.map(
        process_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return ds.batch(
        BATCH_SIZE
    ).prefetch(
        tf.data.AUTOTUNE
    )


# Output helpers

def save_confusion_matrix_png(
    cm,
    class_names,
    output_path,
):
    plt.figure(
        figsize=(9, 7)
    )

    plt.imshow(cm)

    plt.title(
        "Round 9 Held-Out Test - ResNet50"
    )

    plt.xlabel(
        "Predicted class"
    )

    plt.ylabel(
        "True class"
    )

    short_names = [
        n.replace(" Plastic", "")
        for n in class_names
    ]

    plt.xticks(
        range(len(class_names)),
        short_names,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(class_names)),
        short_names,
    )

    plt.colorbar()

    for r in range(len(class_names)):
        for c in range(len(class_names)):
            plt.text(
                c,
                r,
                str(cm[r, c]),
                ha="center",
                va="center",
            )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def log_artifact(
    path,
    artifact_path,
):
    if Path(path).exists():
        mlflow.log_artifact(
            str(path),
            artifact_path=artifact_path,
        )


# Main prediction workflow

def main():
    if not SPLIT_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {SPLIT_MANIFEST_PATH}\n"
            "Run a Round 9 training script first."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_PATH}\n"
            "Run the Round 9 ResNet50 training script first."
        )

    split_df = pd.read_csv(
        SPLIT_MANIFEST_PATH
    )

    test_df = split_df[
        split_df["split"] == "test"
    ].copy()

    if test_df.empty:
        raise ValueError(
            "No held-out test records found in the split manifest."
        )

    missing = [
        p
        for p in test_df["path"].tolist()
        if not Path(p).exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing test image: {missing[0]}"
        )

    test_paths = test_df[
        "path"
    ].tolist()

    test_labels = test_df[
        "label"
    ].astype(
        int
    ).tolist()

    test_ds = create_dataset(
        test_paths,
        test_labels,
    )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "ResNet50Preprocess": ResNet50Preprocess
        },
    )

    with mlflow.start_run(
        run_name="Round_9_ResNet50_held_out_test_prediction"
    ):
        mlflow.set_tags({
            "project": "PlastiSort AI",
            "round": "Round 9",
            "model": MODEL_NAME,
            "task": "held_out_test_prediction",
            "split_strategy": "grouped_stratified_70_20_10",
        })

        mlflow.log_params({
            "model_name": MODEL_NAME,
            "prediction_dataset": "Round 9 held-out test split",
            "test_images": len(test_paths),
            "batch_size": BATCH_SIZE,
            "image_size": "224x224",
            "model_path": str(MODEL_PATH),
            "split_manifest": str(SPLIT_MANIFEST_PATH),
        })

        true_labels = np.array(
            test_labels
        )

        probabilities = model.predict(
            test_ds,
            verbose=0,
        )

        predicted_labels = np.argmax(
            probabilities,
            axis=1,
        )

        confidences = np.max(
            probabilities,
            axis=1,
        )

        accuracy = accuracy_score(
            true_labels,
            predicted_labels,
        )

        precision = precision_score(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        )

        report_text = classification_report(
            true_labels,
            predicted_labels,
            labels=list(
                range(len(VALID_CLASSES))
            ),
            target_names=VALID_CLASSES,
            digits=4,
            zero_division=0,
        )

        report_dict = classification_report(
            true_labels,
            predicted_labels,
            labels=list(
                range(len(VALID_CLASSES))
            ),
            target_names=VALID_CLASSES,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(
            true_labels,
            predicted_labels,
            labels=list(
                range(len(VALID_CLASSES))
            ),
        )

        predictions_path = (
            OUTPUT_DIR
            / "prediction_results_ResNet50_round9_test.csv"
        )

        report_txt_path = (
            OUTPUT_DIR
            / "classification_report_ResNet50_round9_test.txt"
        )

        report_csv_path = (
            OUTPUT_DIR
            / "classification_report_ResNet50_round9_test.csv"
        )

        cm_csv_path = (
            OUTPUT_DIR
            / "confusion_matrix_ResNet50_round9_test.csv"
        )

        cm_png_path = (
            OUTPUT_DIR
            / "confusion_matrix_ResNet50_round9_test.png"
        )

        summary_path = (
            OUTPUT_DIR
            / "prediction_summary_ResNet50_round9_test.csv"
        )

        metrics_json_path = (
            OUTPUT_DIR
            / "prediction_metrics_ResNet50_round9_test.json"
        )

        prediction_data = {
            "image_path": test_paths,
            "true_label": true_labels,
            "true_class": [
                VALID_CLASSES[i]
                for i in true_labels
            ],
            "predicted_label": predicted_labels,
            "predicted_class": [
                VALID_CLASSES[i]
                for i in predicted_labels
            ],
            "confidence": confidences,
        }

        for i, cls in enumerate(
            VALID_CLASSES
        ):
            prediction_data[
                f"probability_{cls.replace(' ', '_')}"
            ] = probabilities[:, i]

        pd.DataFrame(
            prediction_data
        ).to_csv(
            predictions_path,
            index=False,
        )

        report_txt_path.write_text(
            report_text
        )

        pd.DataFrame(
            report_dict
        ).transpose().to_csv(
            report_csv_path
        )

        pd.DataFrame(
            cm,
            index=VALID_CLASSES,
            columns=VALID_CLASSES,
        ).to_csv(
            cm_csv_path
        )

        save_confusion_matrix_png(
            cm,
            VALID_CLASSES,
            cm_png_path,
        )

        pd.DataFrame([{
            "round": "Round_9",
            "model": MODEL_NAME,
            "prediction_set": "held_out_test",
            "test_images": len(test_paths),
            "accuracy": accuracy,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
        }]).to_csv(
            summary_path,
            index=False,
        )

        metrics_json_path.write_text(
            json.dumps(
                {
                    "accuracy": float(accuracy),
                    "precision_weighted": float(precision),
                    "recall_weighted": float(recall),
                    "f1_weighted": float(f1),
                    "test_images": int(len(test_paths)),
                },
                indent=4,
            )
        )

        mlflow.log_metrics({
            "prediction_test_accuracy":
                float(accuracy),

            "prediction_test_precision_weighted":
                float(precision),

            "prediction_test_recall_weighted":
                float(recall),

            "prediction_test_f1_weighted":
                float(f1),
        })

        for path, folder in [
            (
                Path(__file__),
                "source",
            ),
            (
                SPLIT_MANIFEST_PATH,
                "dataset_split",
            ),
            (
                MODEL_PATH,
                "model",
            ),
            (
                predictions_path,
                "held_out_test_prediction",
            ),
            (
                report_txt_path,
                "held_out_test_prediction",
            ),
            (
                report_csv_path,
                "held_out_test_prediction",
            ),
            (
                cm_csv_path,
                "held_out_test_prediction",
            ),
            (
                cm_png_path,
                "held_out_test_prediction",
            ),
            (
                summary_path,
                "held_out_test_prediction",
            ),
            (
                metrics_json_path,
                "held_out_test_prediction",
            ),
        ]:
            log_artifact(
                path,
                folder,
            )

        print("\n" + "=" * 72)
        print(
            "ROUND 9 ResNet50 HELD-OUT TEST "
            "PREDICTION COMPLETED"
        )
        print("=" * 72)

        print(
            f"Test images : {len(test_paths)}"
        )

        print(
            f"Accuracy    : {accuracy * 100:.2f}%"
        )

        print(
            f"Precision   : {precision * 100:.2f}%"
        )

        print(
            f"Recall      : {recall * 100:.2f}%"
        )

        print(
            f"F1 score    : {f1 * 100:.2f}%"
        )

        print(
            f"Output folder: {OUTPUT_DIR}"
        )

        print(
            "\nClassification Report:"
        )

        print(
            report_text
        )


if __name__ == "__main__":
    main()
