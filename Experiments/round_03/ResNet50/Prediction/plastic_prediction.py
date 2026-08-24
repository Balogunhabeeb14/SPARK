# ============================================================
# PlastiSort AI — Round 11 ResNet50 Held-Out Test Prediction
# with MLflow
#
# Round 11 changes from Round 10:
# - Uses Round 11 split manifest and trained model
# - Updated experiment, run name and output paths to Round 11
# ============================================================

import json
from pathlib import Path

# Set matplotlib backend to 'Agg' to allow headless generation of plots (no GUI required)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME   = "ResNet50"
PREDICT_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = PREDICT_DIR.parent

IMG_SIZE   = (224, 224)    # Standard input dimension expected by ResNet50
BATCH_SIZE = 64          # Number of test images processed in parallel per batch

# The five specific plastic categories being evaluated
VALID_CLASSES = [
    "HDPE Plastic", "LDPE Plastic", "PET Plastic",
    "PP Plastic", "PS Plastic",
]

# File and directory paths for loading data/models and saving outputs
SPLIT_MANIFEST_PATH    = PROJECT_ROOT / "round11_grouped_split_70_20_10.csv"
MODEL_PATH             = PROJECT_ROOT / "Train" / "plastic_model_ResNet50_round11.keras"
OUTPUT_DIR             = PROJECT_ROOT / "outputs" / "round11_ResNet50_prediction_outputs"
MLFLOW_TRACKING_DIR    = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT_NAME = "PlastiSort_Round_11_70_20_10"

# Ensure output directories exist before writing files
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)

# Configure MLflow tracking URI and active experiment group
mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_DIR.resolve()}")
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


# ============================================================
# CUSTOM LAYER — required to load ResNet50 model
# ============================================================

@tf.keras.utils.register_keras_serializable(package="PlastiSort")
class ResNet50Preprocess(layers.Layer):
    """
    Custom TensorFlow layer necessary to correctly load the saved model 
    by reapplying official ResNet50 normalization preprocessing.
    """
    def call(self, inputs):
        return tf.keras.applications.resnet50.preprocess_input(
            inputs
        )


# ============================================================
# IMAGE PIPELINE
# ============================================================

def process_image(path, label):
    """Reads an image file from disk, resizes it, and formats it as a float32 tensor."""
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    image.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return image, label


def create_dataset(paths, labels):
    """Builds an optimized tf.data dataset pipeline for batching and prefetching test images."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ============================================================
# ARTIFACT HELPERS
# ============================================================

def save_confusion_matrix_png(cm, class_names, output_path):
    """Generates, styles, and saves a visual confusion matrix plot as a PNG file."""
    plt.figure(figsize=(9, 7))
    plt.imshow(cm)
    plt.title("Round 11 Held-Out Test — ResNet50")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    
    # Shorten class labels for cleaner readability on plot axes
    short_names = [n.replace(" Plastic", "") for n in class_names]
    plt.xticks(
        range(len(class_names)), short_names,
        rotation=45, ha="right"
    )
    plt.yticks(range(len(class_names)), short_names)
    plt.colorbar()
    
    # Annotate individual grid cells with absolute sample counts
    for r in range(len(class_names)):
        for c in range(len(class_names)):
            plt.text(
                c, r, str(cm[r, c]),
                ha="center", va="center"
            )
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def log_artifact(path, artifact_path):
    """Safely logs a local file as an MLflow artifact if it exists on disk."""
    if Path(path).exists():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


# ============================================================
# MAIN
# ============================================================

def main():
    # Verify that the required split file exists
    if not SPLIT_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {SPLIT_MANIFEST_PATH}\n"
            "Run the Round 11 training script first."
        )

    # Verify that the trained model file exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_PATH}\n"
            "Run the Round 11 ResNet50 training script first."
        )

    # Load dataset split and isolate records assigned to the 'test' group
    split_df = pd.read_csv(SPLIT_MANIFEST_PATH)
    test_df  = split_df[split_df["split"] == "test"].copy()

    if test_df.empty:
        raise ValueError(
            "No held-out test records found in the split manifest."
        )

    # Check for missing test images on disk
    missing = [
        p for p in test_df["path"].tolist()
        if not Path(p).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing test image: {missing[0]}")

    # Extract paths and label arrays for evaluation
    test_paths  = test_df["path"].tolist()
    test_labels = test_df["label"].astype(int).tolist()
    test_ds     = create_dataset(test_paths, test_labels)

    # Load the trained model using the custom layer definition
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"ResNet50Preprocess": ResNet50Preprocess}
    )

    # Begin tracking the evaluation run in MLflow
    with mlflow.start_run(
        run_name="Round_11_ResNet50_held_out_test_prediction"
    ):
        # Log metadata tags for the run
        mlflow.set_tags({
            "project": "PlastiSort AI",
            "round": "Round 11",
            "model": MODEL_NAME,
            "task": "held_out_test_prediction",
            "split_strategy": "grouped_stratified_70_20_10",
        })
        
        # Log evaluation configuration parameters
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "prediction_dataset": "Round 11 held-out test split",
            "test_images": len(test_paths),
            "batch_size": BATCH_SIZE,
            "image_size": "224x224",
            "model_path": str(MODEL_PATH),
            "split_manifest": str(SPLIT_MANIFEST_PATH),
        })

        # Run predictions across the test dataset
        true_labels     = np.array(test_labels)
        probabilities   = model.predict(test_ds, verbose=0)
        predicted_labels = np.argmax(probabilities, axis=1)
        confidences     = np.max(probabilities, axis=1)

        # Compute core performance metrics
        accuracy  = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(
            true_labels, predicted_labels,
            average="weighted", zero_division=0
        )
        recall    = recall_score(
            true_labels, predicted_labels,
            average="weighted", zero_division=0
        )
        f1        = f1_score(
            true_labels, predicted_labels,
            average="weighted", zero_division=0
        )

        # Generate detailed text and dictionary classification reports and confusion matrix
        report_text = classification_report(
            true_labels, predicted_labels,
            labels=list(range(len(VALID_CLASSES))),
            target_names=VALID_CLASSES,
            digits=4, zero_division=0,
        )
        report_dict = classification_report(
            true_labels, predicted_labels,
            labels=list(range(len(VALID_CLASSES))),
            target_names=VALID_CLASSES,
            output_dict=True, zero_division=0,
        )
        cm = confusion_matrix(
            true_labels, predicted_labels,
            labels=list(range(len(VALID_CLASSES))),
        )

        # Define file paths for output artifacts
        predictions_path  = OUTPUT_DIR / "prediction_results_ResNet50_round11_test.csv"
        report_txt_path   = OUTPUT_DIR / "classification_report_ResNet50_round11_test.txt"
        report_csv_path   = OUTPUT_DIR / "classification_report_ResNet50_round11_test.csv"
        cm_csv_path       = OUTPUT_DIR / "confusion_matrix_ResNet50_round11_test.csv"
        cm_png_path       = OUTPUT_DIR / "confusion_matrix_ResNet50_round11_test.png"
        summary_path      = OUTPUT_DIR / "prediction_summary_ResNet50_round11_test.csv"
        metrics_json_path = OUTPUT_DIR / "prediction_metrics_ResNet50_round11_test.json"

        # Construct data structure containing granular individual predictions
        prediction_data = {
            "image_path":      test_paths,
            "true_label":      true_labels,
            "true_class":      [VALID_CLASSES[i] for i in true_labels],
            "predicted_label": predicted_labels,
            "predicted_class": [VALID_CLASSES[i] for i in predicted_labels],
            "confidence":      confidences,
        }
        for i, cls in enumerate(VALID_CLASSES):
            prediction_data[
                f"probability_{cls.replace(' ', '_')}"
            ] = probabilities[:, i]

        # Write evaluation reports, summary tables, and image files to disk
        pd.DataFrame(prediction_data).to_csv(predictions_path, index=False)
        report_txt_path.write_text(report_text)
        pd.DataFrame(report_dict).transpose().to_csv(report_csv_path)
        pd.DataFrame(
            cm, index=VALID_CLASSES, columns=VALID_CLASSES
        ).to_csv(cm_csv_path)
        save_confusion_matrix_png(cm, VALID_CLASSES, cm_png_path)

        pd.DataFrame([{
            "round": "Round_11",
            "model": MODEL_NAME,
            "prediction_set": "held_out_test",
            "test_images": len(test_paths),
            "accuracy": accuracy,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
        }]).to_csv(summary_path, index=False)

        metrics_json_path.write_text(json.dumps({
            "accuracy": float(accuracy),
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_weighted": float(f1),
            "test_images": int(len(test_paths)),
        }, indent=4))

        # Log overarching performance metrics to MLflow
        mlflow.log_metrics({
            "prediction_test_accuracy":            float(accuracy),
            "prediction_test_precision_weighted": float(precision),
            "prediction_test_recall_weighted":    float(recall),
            "prediction_test_f1_weighted":        float(f1),
        })

        # Register all generated files as tracked MLflow artifacts
        for path, folder in [
            (Path(__file__),      "source"),
            (SPLIT_MANIFEST_PATH, "dataset_split"),
            (MODEL_PATH,          "model"),
            (predictions_path,    "held_out_test_prediction"),
            (report_txt_path,     "held_out_test_prediction"),
            (report_csv_path,     "held_out_test_prediction"),
            (cm_csv_path,         "held_out_test_prediction"),
            (cm_png_path,         "held_out_test_prediction"),
            (summary_path,        "held_out_test_prediction"),
            (metrics_json_path,   "held_out_test_prediction"),
        ]:
            log_artifact(path, folder)

        # Print final console performance report summary
        print("\n" + "=" * 72)
        print("ROUND 11 ResNet50 HELD-OUT TEST PREDICTION COMPLETED")
        print("=" * 72)
        print(f"Test images : {len(test_paths)}")
        print(f"Accuracy    : {accuracy * 100:.2f}%")
        print(f"Precision   : {precision * 100:.2f}%")
        print(f"Recall      : {recall * 100:.2f}%")
        print(f"F1 score    : {f1 * 100:.2f}%")
        print(f"Output folder: {OUTPUT_DIR}")
        print("\nClassification Report:")
        print(report_text)


if __name__ == "__main__":
    main()
