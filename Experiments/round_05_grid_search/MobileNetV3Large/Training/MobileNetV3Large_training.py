# Round 5 - 72 Combination Hyperparameter Tuning (MobileNetV3Large)
import os

# 1. Force CPU mode to bypass corrupted Apple Metal graphics driver loops
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 2. MUST BE FIRST IMPORT! Prevents memory linkage deadlocks on macOS
print("Initializing hyperparameter search environment for MobileNetV3Large", flush=True)
import tensorflow as tf

print("Loading Machine Learning frameworks (MLflow, Pandas, NumPy, Scikit-Learn)", flush=True)
import json
from pathlib import Path
import mlflow
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

print("System libraries and neural network components loaded successfully", flush=True)

# Basic settings for the project
MODEL_NAME    = "MobileNetV3Large_Round5_Hyperparameter_Tuning"
BATCH_SIZE    = 64
IMG_SIZE      = (224, 224)
VALID_CLASSES = ["HDPE Plastic", "LDPE Plastic", "PET Plastic", "PP Plastic", "PS Plastic", "Unknown"]
NUM_CLASSES   = len(VALID_CLASSES)
TUNING_EPOCHS = 5  # Number of training loops per trial
# Reproducibility

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed(SEED)

# Class weights for 6 classes (including Unknown at index 5)
CLASS_WEIGHTS = {
    0: 0.9365,  # HDPE Plastic
    1: 0.9236,  # LDPE Plastic
    2: 0.9983,  # PET Plastic
    3: 1.1844,  # PP Plastic (Boosted weight)
    4: 1.7326,  # PS Plastic (Boosted weight)
    5: 0.9500   # Unknown Class
}

# Hyperparameter tuning options for testing (2 x 3 x 3 x 4 = 72 Trials)
GRID_OPTIMIZERS     = ["adam", "rmsprop"]
GRID_LEARNING_RATES = [1e-3, 1e-4, 5e-5]
GRID_DROPOUT_RATES   = [0.2, 0.3, 0.5]
UNFREEZE_LAYERS      = [0, 10, 20, 40]

# Looking for the dataset split manifest CSV
PREDICT_DIR   = Path(__file__).resolve().parent
DOWNLOADS_DIR = PREDICT_DIR.parent

manifest_lookups = [
    PREDICT_DIR / "round5_grouped_split_70_20_10.csv",
    DOWNLOADS_DIR / "round5_grouped_split_70_20_10.csv",
    PREDICT_DIR / "ai_plastic_waste_management_training_updated" / "round5_grouped_split_70_20_10.csv"
]
SPLIT_MANIFEST_PATH = next((p for p in manifest_lookups if p.exists()), None)

# Setting up where to save output files and MLflow logs
PROJECT_ROOT = PREDICT_DIR if SPLIT_MANIFEST_PATH is None else SPLIT_MANIFEST_PATH.parent
OUTPUT_DIR    = PROJECT_ROOT / "outputs" / "round5_MobileNetV3Large_tuning_outputs"
MLFLOW_TRACKING_DIR    = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT_NAME = "PlastiSort_Round_5_Hyperparameter_Tuning"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)

# Connecting to MLflow tracking system
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_DIR.resolve()}")
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


# Custom preprocessing layer matching MobileNetV3 requirements
@tf.keras.utils.register_keras_serializable(package="PlastiSort")
class MobileNetV3Preprocess(layers.Layer):
    def call(self, inputs):
        return tf.keras.applications.mobilenet_v3.preprocess_input(inputs)


# macOS-Safe Pure Python Data Generator
class PlastiSortDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, split_name, batch_size=64, img_size=(224, 224), num_classes=NUM_CLASSES, shuffle=False):
        self.df = df[df["split"] == split_name].reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.split_name = split_name
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]
        
        batch_images = []
        batch_labels = []
        
        for _, row in batch_df.iterrows():
            try:
                img = tf.keras.utils.load_img(row["path"], target_size=self.img_size)
                img_array = tf.keras.utils.img_to_array(img)
                
                one_hot_label = np.zeros(self.num_classes, dtype=np.float32)
                one_hot_label[int(row["label"])] = 1.0
                
                batch_images.append(img_array)
                batch_labels.append(one_hot_label)
            except Exception:
                continue
                
        if len(batch_images) == 0:
            return np.empty((0, *self.img_size, 3), dtype=np.float32), np.empty((0, self.num_classes), dtype=np.float32)
            
        return np.array(batch_images, dtype=np.float32), np.array(batch_labels, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_dataset(df, split_name, shuffle=False):
    filtered_df = df[df["split"] == split_name].copy()
    paths = filtered_df["path"].tolist()
    labels = filtered_df["label"].astype(int).tolist()
    
    generator = PlastiSortDataGenerator(df, split_name, batch_size=BATCH_SIZE, img_size=IMG_SIZE, num_classes=NUM_CLASSES, shuffle=shuffle)
    return generator, paths, labels


# Building the neural network with fine-tuning layer depth support
def build_tunable_model(dropout_rate, learning_rate, optimizer_name, unfreeze_layers):
    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )

    if unfreeze_layers == 0:
        # Freeze entire backbone for standard feature extraction
        base_model.trainable = False
    else:
        # Unfreeze only the top N specified layers
        base_model.trainable = True
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = MobileNetV3Preprocess()(inputs)
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    if optimizer_name.lower() == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    if not SPLIT_MANIFEST_PATH:
        print("[ERROR] Missing dataset manifest path configuration", flush=True)
        raise FileNotFoundError("Cannot locate the Round 5 split manifest CSV file.")

    print(f"Targeted Split Manifest: {SPLIT_MANIFEST_PATH}", flush=True)
    print("Reading the dataset manifest file into memory", flush=True)
    split_df = pd.read_csv(SPLIT_MANIFEST_PATH)

    # Print computed dataset class weights for verification
    train_df_labels = split_df[split_df["split"] == "train"]["label"].astype(int).values
    computed_weights = compute_class_weight("balanced", classes=np.unique(train_df_labels), y=train_df_labels)
    dynamic_class_weights = dict(enumerate(computed_weights))
    print(f"Computed Dataset Class Weights from CSV: {dynamic_class_weights}", flush=True)

    print("Building TensorFlow input pipelines for Train, Val, and Test sets", flush=True)
    train_ds, _, _ = create_dataset(split_df, "train", shuffle=True)
    train_eval_ds, train_paths, train_labels = create_dataset(split_df, "train", shuffle=False)
    
    val_split_key = next((s for s in split_df["split"].unique() if s in ["val", "valid", "validation"]), "val")
    val_ds, val_paths, val_labels = create_dataset(split_df, val_split_key, shuffle=False)

    test_split_key = next((s for s in split_df["split"].unique() if s in ["test", "testing"]), "test")
    test_ds, test_paths, test_labels = create_dataset(split_df, test_split_key, shuffle=False)

    total_trials = len(GRID_OPTIMIZERS) * len(GRID_LEARNING_RATES) * len(GRID_DROPOUT_RATES) * len(UNFREEZE_LAYERS)
    leaderboard = []
    trial_idx = 1

    print(f"Starting parent MLflow Grid Search Run (Total Combinations: {total_trials})", flush=True)
    
    with mlflow.start_run(run_name="Round_5_MobileNetV3Large_Grid_Search"):
        mlflow.set_tags({
            "Project": "PlastiSort_AI",
            "Round": "Round_5",
            "Architecture": "MobileNetV3Large",
            "Task": "hyperparameter_tuning_grid_search_round_5"
        })

        # 4-Level Nested Loop: Optimizers x Learning Rates x Dropout Rates x Unfreeze Layers
        for opt_name in GRID_OPTIMIZERS:
            for lr in GRID_LEARNING_RATES:
                for dropout in GRID_DROPOUT_RATES:
                    for unfreeze in UNFREEZE_LAYERS:
                        trial_name = f"Trial_{trial_idx}_{opt_name.upper()}_LR_{lr}_DO_{dropout}_UF_{unfreeze}"
                        print("=" * 70)
                        print(f"Starting combination ({trial_idx}/{total_trials}): {trial_name}")
                        print("=" * 70, flush=True)

                        with mlflow.start_run(run_name=trial_name, nested=True):
                            mlflow.log_params({
                                "optimizer": opt_name,
                                "learning_rate": lr,
                                "dropout_rate": dropout,
                                "unfreeze_layers": unfreeze,
                                "epochs": TUNING_EPOCHS,
                                "batch_size": BATCH_SIZE,
                                "loss_function": "categorical_crossentropy"
                            })

                            model = build_tunable_model(
                                dropout_rate=dropout, 
                                learning_rate=lr, 
                                optimizer_name=opt_name,
                                unfreeze_layers=unfreeze
                            )

                            model.fit(
                                train_ds,
                                validation_data=val_ds,
                                epochs=TUNING_EPOCHS,
                                class_weight=CLASS_WEIGHTS,
                                verbose=1
                            )

                            # --- Evaluation Phase ---
                            print(f"Evaluating Train metrics for {trial_name}...", flush=True)
                            train_probs = model.predict(train_eval_ds, verbose=0)
                            train_preds = np.argmax(train_probs, axis=1)
                            train_acc   = accuracy_score(np.array(train_labels), train_preds)
                            train_f1    = f1_score(np.array(train_labels), train_preds, average="weighted", zero_division=0)

                            print(f"Evaluating Validation metrics for {trial_name}...", flush=True)
                            val_probs = model.predict(val_ds, verbose=0)
                            val_preds = np.argmax(val_probs, axis=1)
                            val_acc   = accuracy_score(np.array(val_labels), val_preds)
                            val_f1    = f1_score(np.array(val_labels), val_preds, average="weighted", zero_division=0)

                            print(f"Evaluating Testing metrics for {trial_name}...", flush=True)
                            test_probs = model.predict(test_ds, verbose=0)
                            test_preds = np.argmax(test_probs, axis=1)
                            test_acc   = accuracy_score(np.array(test_labels), test_preds)
                            test_f1    = f1_score(np.array(test_labels), test_preds, average="weighted", zero_division=0)

                            prec = precision_score(np.array(val_labels), val_preds, average="weighted", zero_division=0)
                            rec  = recall_score(np.array(val_labels), val_preds, average="weighted", zero_division=0)

                            # Log metrics to MLflow dashboard
                            mlflow.log_metrics({
                                "train_accuracy": float(train_acc),
                                "train_f1_weighted": float(train_f1),
                                "val_accuracy": float(val_acc),
                                "val_f1_weighted": float(val_f1),
                                "val_precision_weighted": float(prec),
                                "val_recall_weighted": float(rec),
                                "test_accuracy": float(test_acc),
                                "test_f1_weighted": float(test_f1)
                            })

                            print(f"[{trial_name}] -> Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}% | Test Acc: {test_acc*100:.1f}%", flush=True)

                            trial_model_path = OUTPUT_DIR / f"checkpoint_{trial_name}.keras"
                            model.save(trial_model_path)
                            mlflow.log_artifact(str(trial_model_path), artifact_path="model_checkpoints")

                            leaderboard.append({
                                "trial": trial_name,
                                "optimizer": opt_name,
                                "learning_rate": lr,
                                "dropout_rate": dropout,
                                "unfreeze_layers": unfreeze,
                                "train_accuracy": train_acc,
                                "train_f1_score": train_f1,
                                "val_accuracy": val_acc,
                                "val_f1_score": val_f1,
                                "test_accuracy": test_acc,
                                "test_f1_score": test_f1
                            })

                        trial_idx += 1

        print("Generating comprehensive hyperparameter tuning leaderboard", flush=True)
        leaderboard_df = pd.DataFrame(leaderboard).sort_values(by="val_accuracy", ascending=False)
        
        leaderboard_path = OUTPUT_DIR / "round5_hyperparameter_tuning_leaderboard.csv"
        leaderboard_df.to_csv(leaderboard_path, index=False)
        mlflow.log_artifact(str(leaderboard_path), artifact_path="summary")

        best_trial = leaderboard_df.iloc[0]

        print("\n" + "=" * 80)
        print("Round 5 Hyperparameter Tuning Complete (72 Trials)")
        print("=" * 80)
        print(leaderboard_df[["trial", "learning_rate", "dropout_rate", "unfreeze_layers", "train_accuracy", "val_accuracy", "test_accuracy", "val_f1_score"]].to_string(index=False))
        print("=" * 80)
        print(f"Best Configuration Profile: {best_trial['trial']}")
        print(f"Train Acc: {best_trial['train_accuracy']*100:.2f}% | Val Acc: {best_trial['val_accuracy']*100:.2f}% | Test Acc: {best_trial['test_accuracy']*100:.2f}%")
        print(f"LR: {best_trial['learning_rate']} | Dropout: {best_trial['dropout_rate']} | Unfreeze Layers: {best_trial['unfreeze_layers']}")
        print(f"Leaderboard saved to: {leaderboard_path}")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
