# Round 14 MobileNetV3Large Inference & Evaluation Engine
#Importing all the necessary libraries
import os
import argparse
import json
from pathlib import Path

# Force the script to run on the CPU instead of the GPU. 
# This prevents crashes on Mac computers with Apple Metal graphics drivers.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("Initializing inference environment for MobileNetV3Large...", flush=True)

# Import necessary machine learning and data processing libraries
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

# --- General Settings ---
BATCH_SIZE = 64                  # How many images to process at once
IMG_SIZE = (224, 224)            # The height and width the model expects
VALID_CLASSES = ["HDPE Plastic", "LDPE Plastic", "PET Plastic", "PP Plastic", "PS Plastic", "Unknown"]
NUM_CLASSES = len(VALID_CLASSES) # Total number of categories (6)

# --- File and Folder Paths ---
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent

# List of possible places the dataset CSV file might be saved
manifest_lookups = [
    SCRIPT_DIR / "round14_grouped_split_70_20_10.csv",
    PARENT_DIR / "round14_grouped_split_70_20_10.csv",
    SCRIPT_DIR / "ai_plastic_waste_management_training_updated" / "round14_grouped_split_70_20_10.csv"
]
# Find the first path in the list that actually exists
SPLIT_MANIFEST_PATH = next((p for p in manifest_lookups if p.exists()), None)

# Set up folders for saving the results
PROJECT_ROOT = SCRIPT_DIR if SPLIT_MANIFEST_PATH is None else SPLIT_MANIFEST_PATH.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "round14_MobileNetV3Large_tuning_outputs"
PREDICTION_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"

# Create the prediction output folder if it doesn't already exist
PREDICTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Custom Layers ---
# We must define this custom layer so TensorFlow knows how to load the saved model properly.
# It applies the specific image formatting required by MobileNetV3.
@tf.keras.utils.register_keras_serializable(package="PlastiSort")
class MobileNetV3Preprocess(layers.Layer):
    def call(self, inputs):
        return tf.keras.applications.mobilenet_v3.preprocess_input(inputs)


# --- Data Generator ---
# This class loads images in small batches (e.g., 64 at a time) instead of 
# loading all images into memory at once, which prevents out-of-memory crashes.
class PlastiSortInferenceGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=64, img_size=(224, 224), num_classes=NUM_CLASSES):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes

    # Calculates how many batches make up the entire dataset
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    # Retrieves one batch of images and their labels
    def __getitem__(self, index):
        # Get the rows from the dataframe for the current batch
        batch_df = self.df.iloc[index * self.batch_size : (index + 1) * self.batch_size]
        
        batch_images = []
        batch_labels = []

        # Loop through each row and load the image
        for _, row in batch_df.iterrows():
            try:
                # Load the image and resize it
                img = tf.keras.utils.load_img(row["path"], target_size=self.img_size)
                img_array = tf.keras.utils.img_to_array(img)

                # Create a one-hot encoded array for the label (e.g., [0, 0, 1, 0, 0, 0])
                one_hot_label = np.zeros(self.num_classes, dtype=np.float32)
                one_hot_label[int(row["label"])] = 1.0

                batch_images.append(img_array)
                batch_labels.append(one_hot_label)
            except Exception:
                # If an image fails to load, create a blank (black) image and a zero label
                # This ensures the pipeline doesn't crash halfway through
                batch_images.append(np.zeros((*self.img_size, 3), dtype=np.float32))
                batch_labels.append(np.zeros(self.num_classes, dtype=np.float32))

        # Return the batch as numpy arrays
        return np.array(batch_images, dtype=np.float32), np.array(batch_labels, dtype=np.float32)


# --- Helper Functions ---
def find_best_model_checkpoint():
    """Finds the best model file (.keras) by checking the leaderboard CSV first."""
    leaderboard_path = OUTPUT_DIR / "round14_hyperparameter_tuning_leaderboard.csv"
    
    # If the leaderboard exists, read it to find the best trial name
    if leaderboard_path.exists():
        df_lb = pd.read_csv(leaderboard_path)
        best_trial = df_lb.iloc[0]["trial"] # The top row is the best model
        target_model = OUTPUT_DIR / f"checkpoint_{best_trial}.keras"
        if target_model.exists():
            return target_model

    # If no leaderboard, just find the most recently created .keras file in the folder
    checkpoints = sorted(list(OUTPUT_DIR.glob("*.keras")), key=os.path.getmtime, reverse=True)
    if checkpoints:
        return checkpoints[0]

    return None # Return None if no models are found


# --- Main Prediction Logic ---
def run_prediction(model_path, split_target="test"):
    """Loads a model, runs predictions on a dataset, and saves the results."""
    
    # Ensure we found the dataset CSV
    if not SPLIT_MANIFEST_PATH:
        raise FileNotFoundError("[ERROR] Could not locate round14_grouped_split_70_20_10.csv manifest.")

    print(f"Reading dataset split manifest from: {SPLIT_MANIFEST_PATH}", flush=True)
    split_df = pd.read_csv(SPLIT_MANIFEST_PATH)

    # Check if the requested split (like "test" or "val") actually exists in the CSV
    available_splits = split_df["split"].unique()
    if split_target not in available_splits:
        # If the split isn't found, try to guess the closest match, or pick the last one
        fallback_target = next((s for s in available_splits if split_target in s), available_splits[-1])
        print(f"[WARNING] Split '{split_target}' not found. Using '{fallback_target}' instead.", flush=True)
        split_target = fallback_target

    # Filter the dataset to only include the target split
    target_df = split_df[split_df["split"] == split_target].reset_index(drop=True)
    print(f"Generating predictions for target split: '{split_target}' ({len(target_df)} samples)", flush=True)

    # Set up the data generator
    generator = PlastiSortInferenceGenerator(target_df, batch_size=BATCH_SIZE, img_size=IMG_SIZE)

    # Load the trained model using our custom preprocessing layer
    print(f"Loading MobileNetV3Large Keras Checkpoint: {model_path}", flush=True)
    custom_objects = {"MobileNetV3Preprocess": MobileNetV3Preprocess}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    # --- Run Inference ---
    print("Executing model forward pass predictions...", flush=True)
    raw_probabilities = model.predict(generator, verbose=1)
    
    # Process the results
    # np.argmax gets the index of the highest probability (e.g., 0 for HDPE)
    predicted_indices = np.argmax(raw_probabilities, axis=1)
    # np.max gets the actual probability score (e.g., 0.98)
    confidence_scores = np.max(raw_probabilities, axis=1)
    true_labels = target_df["label"].astype(int).values

    # --- Calculate Evaluation Metrics ---
    acc = accuracy_score(true_labels, predicted_indices)
    prec = precision_score(true_labels, predicted_indices, average="weighted", zero_division=0)
    rec = recall_score(true_labels, predicted_indices, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, predicted_indices, average="weighted", zero_division=0)

    # Print a summary of the performance
    print("\n" + "=" * 80)
    print(f"EVALUATION SUMMARY ({split_target.upper()} SET)")
    print("=" * 80)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")
    print("=" * 80)

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(true_labels, predicted_indices, target_names=VALID_CLASSES, zero_division=0))

    # --- Save Results to CSV ---
    # Add our prediction data back to the original dataframe
    target_df["predicted_label"] = predicted_indices
    target_df["predicted_class"] = [VALID_CLASSES[i] for i in predicted_indices]
    target_df["confidence"] = np.round(confidence_scores, 4)

    # Add columns for the individual probability of each specific class
    for idx, class_name in enumerate(VALID_CLASSES):
        target_df[f"prob_{class_name.replace(' ', '_')}"] = np.round(raw_probabilities[:, idx], 4)

    # Save the updated table to a new CSV file
    output_csv_path = PREDICTION_OUTPUT_DIR / f"predictions_{split_target}_round14.csv"
    target_df.to_csv(output_csv_path, index=False)
    print(f"\nDetailed predictions saved to CSV: {output_csv_path}")

    return target_df


# --- Script Execution Point ---
if __name__ == "__main__":
    # Setup command-line arguments so the user can pass specific settings
    parser = argparse.ArgumentParser(description="PlastiSort Round 14 MobileNetV3Large Prediction Script")
    parser.add_argument("--model_path", type=str, default=None, help="Path to a specific .keras model file")
    parser.add_argument("--split", type=str, default="test", help="Which dataset split to test (test, val, or train)")
    args = parser.parse_args()

    # Determine which model file to use
    if args.model_path:
        # Use the specific file provided in the command line
        model_file = Path(args.model_path)
    else:
        # Otherwise, automatically look for the best one
        model_file = find_best_model_checkpoint()

    # Stop the script if no model could be found
    if model_file is None or not model_file.exists():
        raise FileNotFoundError("[ERROR] Could not find a valid .keras model checkpoint to execute inference.")

    # Start the prediction process
    run_prediction(model_path=model_file, split_target=args.split)
