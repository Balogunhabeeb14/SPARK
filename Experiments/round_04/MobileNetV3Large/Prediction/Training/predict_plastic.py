#Importing all the necessary libraries
import os
import argparse
import numpy as np
import tensorflow as tf

# ==============================================================================
# CUSTOM LAYERS & CONFIGURATION
# ==============================================================================

# Register the exact custom preprocessing layer used during training
# to ensure smooth model loading without deserialization errors.
@tf.keras.utils.register_keras_serializable(package="PlastiSort")
class MobileNetV3Preprocess(tf.keras.layers.Layer):
    def call(self, inputs):
        # Applies MobileNetV3 input scaling (rescales values according to model requirements)
        return tf.keras.applications.mobilenet_v3.preprocess_input(inputs)


# Set paths relative to the directory where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Target input shape for MobileNetV3Large
IMG_SIZE = (224, 224)

# Alphabetically sorted class list matching training labels (0 to 4)
VALID_CLASSES = [
    "HDPE Plastic",
    "LDPE Plastic",
    "PET Plastic",
    "PP Plastic",
    "PS Plastic"
]

# Paths to the saved model artifacts
KERAS_MODEL_PATH = os.path.join(
    BASE_DIR,
    "plastic_model_MobileNetV3Large_round12.keras"
)

TFLITE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "mobilenetv3_plastic_fp16.tflite"
)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_and_preprocess_image(image_path):
    """Loads an image from disk, resizes it, and converts it to a 4D tensor."""
    # Ensure the target image actually exists before processing
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    # Read the raw file bytes from disk
    image = tf.io.read_file(image_path)
    
    # Decode the bytes into a 3-channel RGB image (ignores non-static GIF frames)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    
    # Resize the image to match the target model input size (224x224)
    image = tf.image.resize(image, IMG_SIZE)
    
    # Cast image values to float32
    image = tf.cast(image, tf.float32)
    
    # Expand dims to add batch size -> shape transforms from (224, 224, 3) to (1, 224, 224, 3)
    image = tf.expand_dims(image, axis=0)
    
    return image


def predict_keras(image_tensor, model_path=KERAS_MODEL_PATH):
    """Performs inference using the native Keras (.keras) saved model."""
    print(f"Loading Keras model from: {model_path}")
    
    # Load model and inject custom preprocessing layer mapping
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"MobileNetV3Preprocess": MobileNetV3Preprocess}
    )
    
    # Pass tensor through model with training=False to disable dropout layers
    probabilities = model(image_tensor, training=False).numpy()[0]
    
    return probabilities


def predict_tflite(image_tensor, model_path=TFLITE_MODEL_PATH):
    """Performs inference using the optimized Float16 TFLite model."""
    print(f"Loading TFLite model from: {model_path}")
    
    # Initialize the TFLite runtime interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Retrieve input and output layer tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Pass the preprocessed image tensor directly to the input tensor slot
    interpreter.set_tensor(input_details[0]['index'], image_tensor.numpy())
    
    # Run the inference pass
    interpreter.invoke()

    # Extract class probability output array
    probabilities = interpreter.get_tensor(output_details[0]['index'])[0]
    
    return probabilities


def run_inference(image_path, use_tflite=False):
    """Loads image, runs selected inference model, and formats prediction output."""
    # Step 1: Preprocess target image into a tensor
    image_tensor = load_and_preprocess_image(image_path)
    
    # Step 2: Route tensor to selected prediction model backend
    if use_tflite:
        probs = predict_tflite(image_tensor)
    else:
        probs = predict_keras(image_tensor)

    # Step 3: Extract highest probability score and predicted target class label
    predicted_index = np.argmax(probs)
    predicted_class = VALID_CLASSES[predicted_index]
    confidence = probs[predicted_index] * 100

    # Step 4: Display clean summary to console
    print("\n" + "=" * 50)
    print("PLASTISORT PREDICTION RESULTS")
    print("=" * 50)
    print(f"Target Image     : {os.path.basename(image_path)}")
    print(f"Predicted Class  : {predicted_class}")
    print(f"Confidence Score : {confidence:.2f}%\n")
    print("Class Probabilities Breakdown:")
    print("-" * 50)
    for index, class_name in enumerate(VALID_CLASSES):
        print(f"  {class_name:<15}: {probs[index] * 100:6.2f}%")
    print("=" * 50)

    # Return structured result dictionary for modular code imports
    return {
        "class": predicted_class,
        "confidence": confidence,
        "probabilities": dict(zip(VALID_CLASSES, probs.tolist()))
    }


# ==============================================================================
# COMMAND LINE INTERFACE (CLI)
# ==============================================================================

if __name__ == "__main__":
    # Setup command-line argument parser
    parser = argparse.ArgumentParser(description="PlastiSort Single-Image Prediction Pipeline")
    
    # Required flag: image file path
    parser.add_argument(
        "--image", 
        type=str, 
        required=True, 
        help="Path to target image file for prediction"
    )
    
    # Optional flag: pass --tflite to execute prediction using TFLite runtime instead of Keras
    parser.add_argument(
        "--tflite", 
        action="store_true", 
        help="Use TFLite model instead of full Keras model"
    )

    # Parse args from command line execution
    args = parser.parse_args()
    
    # Run prediction execution pipeline
    run_inference(args.image, use_tflite=args.tflite)
