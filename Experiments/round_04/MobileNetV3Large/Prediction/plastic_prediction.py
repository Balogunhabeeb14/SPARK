#Round 4 - Comprehensive Multi-Split Evaluation (MobileNetV3Large Smart-Split Edition)
print("Initializing multi-split validation environment for MobileNetV3Large...", flush=True)

import json
import os
import csv
from pathlib import Path

print("Loading Machine Learning frameworks (TensorFlow, MLflow, Pandas)...", flush=True)
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

print(" Systems libraries and neural network components loaded successfully.", flush=True)

#Configurations
MODEL_NAME = "MobileNetV3Large_Round4"
BATCH_SIZE = 64
IMG_SIZE   = (224, 224)
VALID_CLASSES = ["HDPE Plastic", "LDPE Plastic", "PET Plastic", "PP Plastic", "PS Plastic"]

#SMART PATH AUTO-DETECTION
PREDICT_DIR  = Path(__file__).resolve().parent
DOWNLOADS_DIR = PREDICT_DIR.parent

manifest_lookups = [
    PREDICT_DIR / "round4_grouped_split_70_20_10.csv",
    DOWNLOADS_DIR / "round4_grouped_split_70_20_10.csv",
    PREDICT_DIR / "ai_plastic_waste_management_training_updated" / "round4_grouped_split_70_20_10.csv"
]
SPLIT_MANIFEST_PATH = next((p for p in manifest_lookups if p.exists()), None)

# Lookups updated to target the MobileNetV3Large weights binary file
model_lookups = [
    PREDICT_DIR / "Train" / "plastic_model_MobileNetV3Large_round4.keras",
    PREDICT_DIR / "plastic_model_MobileNetV3Large_round4.keras",
    DOWNLOADS_DIR / "Train" / "plastic_model_MobileNetV3Large_round4.keras",
    DOWNLOADS_DIR / "plastic_model_MobileNetV3Large_round4.keras"
]
MODEL_PATH = next((p for p in model_lookups if p.exists()), None)

PROJECT_ROOT = PREDICT_DIR if SPLIT_MANIFEST_PATH is None else SPLIT_MANIFEST_PATH.parent
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "round4_MobileNetV3Large_prediction_outputs"
MLFLOW_TRACKING_DIR    = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT_NAME = "PlastiSort_Round_4_70_20_10"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_DIR.resolve()}")
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


#REQUIRED CUSTOM OBJECT REGISTRATION FOR MOBILENETV3 CUSTOM LAYERS
@tf.keras.utils.register_keras_serializable(
    package="PlastiSort"
)
class MobileNetV3Preprocess(layers.Layer):
    def call(self, inputs):
        return tf.keras.applications.mobilenet_v3.preprocess_input(
            inputs
        )
    
#Image processing pipeline for MobileNetV3Large
def process_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    image.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return image, label

def create_dataset(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def log_artifact(path, artifact_path):
    if Path(path).exists():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)

def main():
    if not SPLIT_MANIFEST_PATH or not MODEL_PATH:
        print("[ERROR] Missing training assets path configuration!", flush=True)
        if not SPLIT_MANIFEST_PATH: raise FileNotFoundError("Could not locate the CSV split manifest.")
        if not MODEL_PATH: raise FileNotFoundError("Could not locate the trained MobileNetV3Large .keras model binary.")

    print(f"Auto-Targeted Manifest : {SPLIT_MANIFEST_PATH}", flush=True)
    print(f"Auto-Targeted Model    : {MODEL_PATH}", flush=True)

    print("Reading dataset manifest data file...", flush=True)
    split_df = pd.read_csv(SPLIT_MANIFEST_PATH)

    print(f"Reading compiled Keras model weights file into memory...", flush=True)
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"MobileNetV3Preprocess": MobileNetV3Preprocess}
    )
    print(f"🤖 {MODEL_NAME} network layers and parameters loaded successfully.", flush=True)

    # DYNAMIC SPLIT DETECTION: Scan manifest for whatever name variant is used
    available_splits = split_df["split"].unique().tolist()
    print(f"📋 Detected split labels inside your CSV manifest: {available_splits}", flush=True)
    
    target_splits = []
    if "train" in available_splits: 
        target_splits.append("train")
        
    #Catching any validation alias variation ('val', 'valid', or 'validation')
    val_alias = next((s for s in available_splits if s in ["val", "valid", "validation"]), None)
    if val_alias: 
        target_splits.append(val_alias)
        
    if "test" in available_splits: 
        target_splits.append("test")

    summary_metrics = {}

    print("Initializing parent MLflow multi-split logging run context...", flush=True)
    with mlflow.start_run(run_name="Round_4_MobileNetV3Large_Full_Split_Evaluation"):
        mlflow.set_tags({
            "project": "PlastiSort AI",
            "round": "Round 4",
            "model": MODEL_NAME,
            "task": "complete_split_evaluation",
        })

        for split in target_splits:
            print(f"\n" + "="*60)
            print(f"Processing evaluation slice: {split.upper()} set", flush=True)
            print("="*60)

            filtered_df = split_df[split_df["split"] == split].copy()
            paths = filtered_df["path"].tolist()
            labels = filtered_df["label"].astype(int).tolist()
            
            missing = [p for p in paths if not Path(p).exists()]
            if missing:
                raise FileNotFoundError(f"Missing image file asset in '{split}': {missing[0]}")

            print(f"Building pipeline for {len(paths)} images...", flush=True)
            ds = create_dataset(paths, labels)

            print(f"Computing MobileNetV3Large predictions for the {split.upper()} split:")
            probabilities = model.predict(ds, verbose=1)
            
            predicted_labels = np.argmax(probabilities, axis=1)
            true_labels = np.array(labels)
            confidences = np.max(probabilities, axis=1)

            accuracy  = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)
            recall    = recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)
            f1        = f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)

            summary_metrics[split] = {
                "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "count": len(paths)
            }

            report_text = classification_report(true_labels, predicted_labels, labels=list(range(len(VALID_CLASSES))), target_names=VALID_CLASSES, digits=4, zero_division=0)
            report_dict = classification_report(true_labels, predicted_labels, labels=list(range(len(VALID_CLASSES))), target_names=VALID_CLASSES, output_dict=True, zero_division=0)
            cm = confusion_matrix(true_labels, predicted_labels, labels=list(range(len(VALID_CLASSES))))
            
            #Paths mapped cleanly to MobileNetV3Large naming schemas
            predictions_path  = OUTPUT_DIR / f"prediction_results_MobileNetV3Large_round4_{split}.csv"
            report_txt_path   = OUTPUT_DIR / f"classification_report_MobileNetV3Large_round4_{split}.txt"
            report_csv_path   = OUTPUT_DIR / f"classification_report_MobileNetV3Large_round4_{split}.csv"
            cm_csv_path       = OUTPUT_DIR / f"confusion_matrix_MobileNetV3Large_round4_{split}.csv"

            print(f"Committing performance logs for {split.upper()} to disk...", flush=True)
            prediction_data = {
                "image_path": paths, "true_label": true_labels,
                "true_class": [VALID_CLASSES[i] for i in true_labels],
                "predicted_label": predicted_labels,
                "predicted_class": [VALID_CLASSES[i] for i in predicted_labels],
                "confidence": confidences,
            }
            for i, cls in enumerate(VALID_CLASSES):
                prediction_data[f"probability_{cls.replace(' ', '_')}"] = probabilities[:, i]

            pd.DataFrame(prediction_data).to_csv(predictions_path, index=False)
            report_txt_path.write_text(report_text)
            pd.DataFrame(report_dict).transpose().to_csv(report_csv_path)
            pd.DataFrame(cm, index=VALID_CLASSES, columns=VALID_CLASSES).to_csv(cm_csv_path)

            #Normalizes validation metric keys to 'val' inside MLflow for clean panel tracking graphs
            mlflow_metric_prefix = "val" if split in ["val", "valid", "validation"] else split
            mlflow.log_metrics({
                f"{mlflow_metric_prefix}_mobilenetv3large_accuracy":            float(accuracy),
                f"{mlflow_metric_prefix}_mobilenetv3large_precision_weighted": float(precision),
                f"{mlflow_metric_prefix}_mobilenetv3large_recall_weighted":    float(recall),
                f"{mlflow_metric_prefix}_mobilenetv3large_f1_weighted":        float(f1),
            })

            for file_p in [predictions_path, report_txt_path, report_csv_path, cm_csv_path]:
                log_artifact(file_p, f"mobilenetv3large_split_evaluations/{split}")

        summary_path = OUTPUT_DIR / "mobilenetv3large_cross_split_comparison_summary.csv"
        pd.DataFrame.from_dict(summary_metrics, orient="index").to_csv(summary_path)
        log_artifact(summary_path, "summary")

        print("\n" + "=" * 72)
        print(f"All dataset splits have been successfully evaluated via {MODEL_NAME}!")
        print("=" * 72)
        for split, metrics in summary_metrics.items():
            print(f"{split.upper():<5} Subset ({metrics['count']} imgs) -> Accuracy: {metrics['accuracy'] * 100:.2f}% | F1-Score: {metrics['f1'] * 100:.2f}%")
        print("-" * 72)
        print(f"All MobileNetV3Large reports and text data tables saved to:\n📂 {OUTPUT_DIR}")
        print("=" * 72 + "\n")

if __name__ == "__main__":
    main()
