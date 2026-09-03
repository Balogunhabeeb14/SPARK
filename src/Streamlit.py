"""


Three input modes are available at runtime:

    roi            Classify the full adjustable region of interest.
    roi_with_yolo  Classify the region of interest and draw a detection box.
    yolo           Classify only the detected object crop.


"""

from __future__ import annotations

import io
import os
import threading
import time
from collections import deque
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from ultralytics import YOLO


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.environ.get("PLASTISORT_MODEL", APP_DIR / "models" / "plastic_model.keras")
)

YOLO_MODEL_NAME = os.environ.get("PLASTISORT_YOLO", "yolov8n.pt")

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "HDPE Plastic",
    "LDPE Plastic",
    "PET Plastic",
    "PP Plastic",
    "PS Plastic",
    "UNKNOWN",
]

MODE_LABELS = {
    "roi": "ROI only",
    "roi_with_yolo": "ROI + YOLO rectangle",
    "yolo": "YOLO crop only",
}

MODE_SHORT = {
    "roi": "ROI ONLY",
    "roi_with_yolo": "ROI + YOLO",
    "yolo": "YOLO CROP",
}

ZONE_LABELS = {
    "roi": "ROI CLASSIFICATION ZONE",
    "roi_with_yolo": "CLASSIFICATION + YOLO ZONE",
    "yolo": "YOLO CROP ZONE",
}

# Inference settings that are fixed rather than user-adjustable.
#
# PREPROCESS_MODE  The saved model contains a Rescaling layer
#                  (scale 1/127.5, offset -1.0), so it expects raw 0-255 RGB
#                  and performs the normalisation internally.
# TTA_MODE         "none" classifies one upright view. "mirror" also
#                  classifies a horizontally flipped copy and averages the
#                  two, at roughly double the inference cost.
# UNDO_CLASS_WEIGHTS
#                  Training applied a 1.3x class-weight boost to PP and PS,
#                  which causes those classes to be over-predicted at
#                  inference time. When enabled, predicted probabilities are
#                  divided by the training weights and renormalised.
# BLANK_FRAME_GUARD
#                  Reports NO OBJECT when the crop is almost entirely black,
#                  so an empty scene is not assigned a plastic class. Only
#                  has an effect when background suppression is active.
PREPROCESS_MODE = "raw255"
TTA_MODE = "none"
UNDO_CLASS_WEIGHTS = True
BLANK_FRAME_GUARD = True

# Per-class training image counts, in CLASS_NAMES order. Used to reconstruct
# the class weights that sklearn's compute_class_weight produced.
TRAIN_CLASS_COUNTS = [1118, 1037, 1206, 1099, 997, 772]
PP_PS_WEIGHT_BOOST = 1.3

# Training resized images with tf.image.resize, which is bilinear.
RESIZE_INTERPOLATION = cv2.INTER_LINEAR

BLANK_PIXEL_LEVEL = 30
MIN_CONTENT_RATIO = 0.02

DEFAULT_INPUT_MODE = "roi_with_yolo"
DEFAULT_TEMPORAL_FRAMES = 1
DEFAULT_YOLO_CONFIDENCE = 0.15
DEFAULT_CLASSIFY_EVERY_N = 2
DEFAULT_YOLO_EVERY_N = 5

YOLO_IMAGE_SIZE = 416
YOLO_DEVICE = os.environ.get("PLASTISORT_DEVICE", "cpu")

# Detections of these classes are discarded. Large fixed furniture only;
# classes such as "vase" and "cell phone" are deliberately absent because
# YOLO frequently assigns them to plastic containers and flat plastic items.
IGNORED_YOLO_CLASS_NAMES = {
    "person", "chair", "couch", "bed", "dining table", "toilet",
    "bench", "tv", "refrigerator", "oven", "sink",
}

# When non-empty, only these YOLO classes are considered. Note that plastic
# film, bags and foam are not represented in the COCO label set, so an
# allowlist can make LDPE and PS undetectable.
YOLO_ALLOWED_CLASS_NAMES: set[str] = set()

MIN_DETECTION_AREA_RATIO = 0.002
MAX_DETECTION_AREA_RATIO = 0.90
PAD_RATIO = 0.08

DEFAULT_ROI = (0.13, 0.12, 0.87, 0.92)
ROI_MOVE_STEP = 0.02
ROI_RESIZE_STEP = 0.025
MIN_ROI_SIZE = 0.15


# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------

st.set_page_config(page_title="PlastiSort", layout="wide")

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1500px;
            padding-top: 1rem;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(127, 127, 127, 0.08);
            border: 1px solid rgba(127, 127, 127, 0.18);
            border-radius: 0.75rem;
            padding: 0.75rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("PlastiSort - Live Plastic Recognition")
st.caption(
    "Switch between region-of-interest classification, classification with an "
    "independent detection box, and detection-cropped classification."
)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the six-class classifier...")
def load_plastic_model(model_path: str):
    """Load and validate the saved six-class Keras model."""
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as first_error:
        try:
            model = tf.keras.models.load_model(
                model_path, compile=False, safe_mode=False
            )
        except Exception as second_error:
            raise RuntimeError(
                f"Could not load the classifier.\n\n"
                f"First attempt:\n{first_error}\n\n"
                f"Second attempt:\n{second_error}"
            ) from second_error

    output_count = int(model.output_shape[-1])
    if output_count != len(CLASS_NAMES):
        raise ValueError(
            f"The model produces {output_count} outputs but this application "
            f"expects {len(CLASS_NAMES)}."
        )
    return model


@st.cache_resource(show_spinner="Loading the object detector...")
def load_yolo_model(model_name: str):
    return YOLO(model_name)


if not MODEL_PATH.is_file():
    st.error(
        f"Classifier weights not found at `{MODEL_PATH}`.\n\n"
        "Place the `.keras` file there, or set the `PLASTISORT_MODEL` "
        "environment variable to its location."
    )
    st.stop()

try:
    plastic_model = load_plastic_model(str(MODEL_PATH))
    yolo_model = load_yolo_model(YOLO_MODEL_NAME)
except Exception as error:
    st.exception(error)
    st.stop()


def build_training_class_weights() -> np.ndarray:
    """Reconstruct the class weights applied during training."""
    counts = np.asarray(TRAIN_CLASS_COUNTS, dtype=np.float64)
    weights = counts.sum() / (len(counts) * counts)
    weights[CLASS_NAMES.index("PP Plastic")] *= PP_PS_WEIGHT_BOOST
    weights[CLASS_NAMES.index("PS Plastic")] *= PP_PS_WEIGHT_BOOST
    return weights


TRAINING_CLASS_WEIGHTS = build_training_class_weights()


# --------------------------------------------------------------------------
# Region of interest state
# --------------------------------------------------------------------------

for state_name, default_value in zip(
    ("roi_left", "roi_top", "roi_right", "roi_bottom"), DEFAULT_ROI
):
    if state_name not in st.session_state:
        st.session_state[state_name] = default_value


def set_roi_state(left: float, top: float, right: float, bottom: float) -> None:
    """
    Store a valid fractional region of interest in session state.

    Left and top are clamped to at most 1.0 - MIN_ROI_SIZE, which matches the
    maximum of the corresponding sliders. Without that guarantee, a button
    callback could write a value outside a bound widget's range and Streamlit
    would raise on the next rerun.
    """
    width = min(max(MIN_ROI_SIZE, right - left), 1.0)
    height = min(max(MIN_ROI_SIZE, bottom - top), 1.0)

    left = max(0.0, min(left, 1.0 - width))
    top = max(0.0, min(top, 1.0 - height))

    st.session_state.roi_left = round(float(left), 2)
    st.session_state.roi_top = round(float(top), 2)
    st.session_state.roi_right = round(float(left + width), 2)
    st.session_state.roi_bottom = round(float(top + height), 2)


def move_roi_state(dx: float, dy: float) -> None:
    set_roi_state(
        st.session_state.roi_left + dx,
        st.session_state.roi_top + dy,
        st.session_state.roi_right + dx,
        st.session_state.roi_bottom + dy,
    )


def resize_roi_state(change: float) -> None:
    left, top = st.session_state.roi_left, st.session_state.roi_top
    right, bottom = st.session_state.roi_right, st.session_state.roi_bottom

    centre_x, centre_y = (left + right) / 2.0, (top + bottom) / 2.0
    width = max(MIN_ROI_SIZE, (right - left) + 2.0 * change)
    height = max(MIN_ROI_SIZE, (bottom - top) + 2.0 * change)

    set_roi_state(
        centre_x - width / 2.0,
        centre_y - height / 2.0,
        centre_x + width / 2.0,
        centre_y + height / 2.0,
    )


def reset_roi_state() -> None:
    set_roi_state(*DEFAULT_ROI)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    st.caption(f"Classifier: {MODEL_PATH.name}")
    st.caption(f"Detector: {YOLO_MODEL_NAME}")

    st.subheader("Prediction")

    mode_options = ("roi", "roi_with_yolo", "yolo")
    input_mode = st.selectbox(
        "Input mode",
        options=mode_options,
        index=mode_options.index(DEFAULT_INPUT_MODE),
        format_func=MODE_LABELS.get,
        help=(
            "ROI only classifies the full yellow zone. ROI + YOLO classifies "
            "the same zone and adds a detection box for reference. YOLO crop "
            "classifies only the detected object."
        ),
    )

    temporal_frames = st.slider(
        "Rolling frames",
        1,
        10,
        DEFAULT_TEMPORAL_FRAMES,
        1,
        help="1 reports the current frame with no smoothing.",
    )

    yolo_confidence = st.slider(
        "Detection threshold", 0.05, 0.80, DEFAULT_YOLO_CONFIDENCE, 0.01
    )

    st.subheader("Performance")

    classify_every_n = st.slider(
        "Classify every N frames",
        1,
        10,
        DEFAULT_CLASSIFY_EVERY_N,
        1,
        help="Higher values keep playback smoother and update the label less often.",
    )

    yolo_every_n = st.slider(
        "Detect every N frames",
        1,
        15,
        DEFAULT_YOLO_EVERY_N,
        1,
        help="The detection box is reused between runs.",
    )

    show_probabilities = st.checkbox("Show probability table", value=False)

    st.divider()
    st.subheader("Region of interest")

    up_col, reset_col = st.columns(2)
    up_col.button(
        "Up", use_container_width=True,
        on_click=move_roi_state, args=(0.0, -ROI_MOVE_STEP),
    )
    reset_col.button(
        "Reset", use_container_width=True, on_click=reset_roi_state
    )

    left_col, down_col, right_col = st.columns(3)
    left_col.button(
        "Left", use_container_width=True,
        on_click=move_roi_state, args=(-ROI_MOVE_STEP, 0.0),
    )
    down_col.button(
        "Down", use_container_width=True,
        on_click=move_roi_state, args=(0.0, ROI_MOVE_STEP),
    )
    right_col.button(
        "Right", use_container_width=True,
        on_click=move_roi_state, args=(ROI_MOVE_STEP, 0.0),
    )

    shrink_col, enlarge_col = st.columns(2)
    shrink_col.button(
        "Shrink", use_container_width=True,
        on_click=resize_roi_state, args=(-ROI_RESIZE_STEP,),
    )
    enlarge_col.button(
        "Enlarge", use_container_width=True,
        on_click=resize_roi_state, args=(ROI_RESIZE_STEP,),
    )

    # Slider bounds are derived from MIN_ROI_SIZE so that they always agree
    # with the clamping performed in set_roi_state.
    roi_left = st.slider(
        "Left edge", 0.00, 1.0 - MIN_ROI_SIZE, step=0.01, key="roi_left"
    )
    roi_top = st.slider(
        "Top edge", 0.00, 1.0 - MIN_ROI_SIZE, step=0.01, key="roi_top"
    )
    roi_right = st.slider(
        "Right edge", MIN_ROI_SIZE, 1.00, step=0.01, key="roi_right"
    )
    roi_bottom = st.slider(
        "Bottom edge", MIN_ROI_SIZE, 1.00, step=0.01, key="roi_bottom"
    )

roi_is_valid = roi_right > roi_left and roi_bottom > roi_top
if not roi_is_valid:
    st.warning(
        "The right edge must exceed the left edge and the bottom edge must "
        "exceed the top edge. Use Reset to restore a valid region."
    )


# --------------------------------------------------------------------------
# Video processor
# --------------------------------------------------------------------------

class PlasticVideoProcessor(VideoProcessorBase):
    """
    Classify frames from a browser camera stream.

    The lock protects only short reads and writes of shared state. Model
    inference runs outside it, because get_status is called from the
    Streamlit thread on every rerun and would otherwise block on inference.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()

        self.plastic_model = plastic_model
        self.yolo_model = yolo_model

        self.input_mode = input_mode
        self.temporal_frames = int(temporal_frames)
        self.yolo_threshold = float(yolo_confidence)
        self.classify_every_n = int(classify_every_n)
        self.yolo_every_n = int(yolo_every_n)
        self.roi_fractions = (roi_left, roi_top, roi_right, roi_bottom)

        # Owned by the video thread.
        self.temporal_history: deque = deque(maxlen=max(1, self.temporal_frames))
        self.frame_index = 0
        self.cached_box = None
        self.pending_history_reset = False
        self.pending_history_maxlen = None

        # Shared results, guarded by the lock.
        self.last_prediction = "Waiting..."
        self.last_confidence = 0.0
        self.last_probabilities = None
        self.last_detection_found = False
        self.last_yolo_error = None

        self.last_display_rgb = None
        self.last_classifier_rgb = None
        self.last_input_rgb = None

        self.fps = 0.0
        self.fps_counter = 0
        self.fps_start_time = time.perf_counter()

    # -- settings ----------------------------------------------------------

    def update_settings(
        self,
        input_mode_value: str,
        temporal_value: int,
        yolo_threshold_value: float,
        classify_every_value: int,
        yolo_every_value: int,
        roi_values: tuple,
    ) -> None:
        """Apply sidebar changes to the running video thread."""
        with self.lock:
            mode_changed = input_mode_value != self.input_mode
            pipeline_changed = (
                mode_changed or tuple(roi_values) != self.roi_fractions
            )

            self.input_mode = input_mode_value
            self.yolo_threshold = float(yolo_threshold_value)
            self.classify_every_n = max(1, int(classify_every_value))
            self.yolo_every_n = max(1, int(yolo_every_value))
            self.roi_fractions = tuple(roi_values)

            # The video thread owns the deque, so request a rebuild rather
            # than replacing the object from another thread.
            if int(temporal_value) != self.temporal_frames:
                self.temporal_frames = int(temporal_value)
                self.pending_history_maxlen = max(1, self.temporal_frames)
            elif pipeline_changed:
                self.pending_history_reset = True

            if mode_changed:
                self.cached_box = None
                self.last_prediction = "Waiting..."
                self.last_confidence = 0.0
                self.last_probabilities = None
                self.last_classifier_rgb = None
                self.last_input_rgb = None

    def read_settings(self) -> dict:
        """Copy the settings the video thread needs, then release the lock."""
        with self.lock:
            return {
                "input_mode": self.input_mode,
                "yolo_threshold": self.yolo_threshold,
                "classify_every_n": self.classify_every_n,
                "yolo_every_n": self.yolo_every_n,
                "roi_fractions": self.roi_fractions,
                "history_reset": self.pending_history_reset,
                "history_maxlen": self.pending_history_maxlen,
            }

    def clear_pending_history_flags(self) -> None:
        with self.lock:
            self.pending_history_reset = False
            self.pending_history_maxlen = None

    # -- geometry ----------------------------------------------------------

    @staticmethod
    def roi_bounds_for(frame: np.ndarray, fractions: tuple) -> tuple:
        height, width = frame.shape[:2]
        left, top, right, bottom = fractions

        x1 = max(0, min(int(width * left), width - 1))
        y1 = max(0, min(int(height * top), height - 1))
        x2 = max(x1 + 1, min(int(width * right), width))
        y2 = max(y1 + 1, min(int(height * bottom), height))
        return x1, y1, x2, y2

    # -- detection ---------------------------------------------------------

    def detect_best_object(
        self, rgb_frame: np.ndarray, roi_box: tuple, threshold: float
    ):
        """
        Return the highest-scoring detection inside the region of interest,
        in full-frame coordinates, or None.

        Candidates are ranked by confidence, proximity to the centre of the
        region, and area, so that a centred object is preferred over
        peripheral background clutter.
        """
        roi_x1, roi_y1, roi_x2, roi_y2 = roi_box
        roi_rgb = rgb_frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi_rgb.size == 0:
            return None

        roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
        results = self.yolo_model.predict(
            source=roi_bgr,
            conf=threshold,
            imgsz=YOLO_IMAGE_SIZE,
            device=YOLO_DEVICE,
            verbose=False,
        )

        roi_height, roi_width = roi_rgb.shape[:2]
        roi_area = float(max(1, roi_height * roi_width))
        candidates = []

        for result in results:
            if result.boxes is None:
                continue
            names = result.names
            for box in result.boxes:
                class_name = str(names[int(box.cls[0].item())]).lower()
                if class_name in IGNORED_YOLO_CLASS_NAMES:
                    continue
                if (
                    YOLO_ALLOWED_CLASS_NAMES
                    and class_name not in YOLO_ALLOWED_CLASS_NAMES
                ):
                    continue

                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = (
                    float(v) for v in box.xyxy[0].detach().cpu().numpy()
                )

                box_width = max(0.0, x2 - x1)
                box_height = max(0.0, y2 - y1)
                area_ratio = (box_width * box_height) / roi_area

                if not (
                    MIN_DETECTION_AREA_RATIO
                    <= area_ratio
                    <= MAX_DETECTION_AREA_RATIO
                ):
                    continue

                centre_x = (x1 + x2) / 2.0
                centre_y = (y1 + y2) / 2.0
                distance = np.sqrt(
                    ((centre_x - roi_width / 2.0) / max(1.0, roi_width)) ** 2
                    + ((centre_y - roi_height / 2.0) / max(1.0, roi_height)) ** 2
                )

                score = (
                    confidence
                    * max(0.20, 1.0 - distance)
                    * np.sqrt(max(area_ratio, 1e-8))
                )
                candidates.append((score, x1, y1, x2, y2))

        if not candidates:
            return None

        _, x1, y1, x2, y2 = max(candidates, key=lambda item: item[0])

        pad_x = (x2 - x1) * PAD_RATIO
        pad_y = (y2 - y1) * PAD_RATIO

        left = int(max(roi_x1, roi_x1 + x1 - pad_x))
        top = int(max(roi_y1, roi_y1 + y1 - pad_y))
        right = int(min(roi_x2, roi_x1 + x2 + pad_x))
        bottom = int(min(roi_y2, roi_y1 + y2 + pad_y))

        if right <= left or bottom <= top:
            return None
        return left, top, right, bottom

    # -- classification ----------------------------------------------------

    @staticmethod
    def apply_preprocess(rgb_float: np.ndarray) -> np.ndarray:
        if PREPROCESS_MODE == "raw255":
            return rgb_float
        if PREPROCESS_MODE == "unit":
            return rgb_float / 255.0
        if PREPROCESS_MODE == "centred":
            return (rgb_float / 127.5) - 1.0
        raise ValueError(f"Unknown preprocessing mode: {PREPROCESS_MODE}")

    def build_batch(self, crop_rgb: np.ndarray):
        """Build the classifier input batch from a crop."""
        views = [crop_rgb]
        if TTA_MODE == "mirror":
            views.append(np.ascontiguousarray(crop_rgb[:, ::-1]))
        elif TTA_MODE != "none":
            raise ValueError(f"Unknown TTA mode: {TTA_MODE}")

        prepared = []
        for view in views:
            resized = cv2.resize(
                np.ascontiguousarray(view),
                IMG_SIZE,
                interpolation=RESIZE_INTERPOLATION,
            )
            prepared.append(resized.astype(np.float32))

        batch = np.stack([self.apply_preprocess(v) for v in prepared], axis=0)
        return batch, prepared[0].astype(np.uint8)

    @staticmethod
    def content_ratio(rgb_crop) -> float:
        """Fraction of pixels that are not close to black."""
        if rgb_crop is None or rgb_crop.size == 0:
            return 0.0
        return float(np.mean(rgb_crop.max(axis=2) > BLANK_PIXEL_LEVEL))

    def classify(self, crop_rgb: np.ndarray):
        """Run the classifier. Called without the lock held."""
        batch, model_input = self.build_batch(crop_rgb)
        view_probabilities = np.asarray(
            self.plastic_model(batch, training=False)
        )

        frame_probabilities = (
            view_probabilities[0]
            if view_probabilities.shape[0] == 1
            else np.mean(view_probabilities, axis=0)
        )

        self.temporal_history.append(frame_probabilities)

        probabilities = (
            frame_probabilities
            if len(self.temporal_history) == 1
            else np.mean(np.stack(list(self.temporal_history), axis=0), axis=0)
        )

        if UNDO_CLASS_WEIGHTS:
            probabilities = probabilities / TRAINING_CLASS_WEIGHTS
            probabilities = probabilities / max(
                float(probabilities.sum()), 1e-9
            )

        index = int(np.argmax(probabilities))
        return (
            CLASS_NAMES[index],
            float(probabilities[index]),
            probabilities,
            model_input,
        )

    # -- drawing -----------------------------------------------------------

    @staticmethod
    def draw_rectangle(frame, box, colour, thickness) -> None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

    def draw_panel(self, frame: np.ndarray, mode: str) -> None:
        """Draw the result panel in the top-right corner of the frame."""
        height, width = frame.shape[:2]
        panel_width = min(440, width - 30)
        x1 = max(10, width - panel_width - 15)
        y1 = 15
        x2 = width - 15
        y2 = min(height - 10, y1 + 142)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        if self.last_prediction == "UNKNOWN":
            colour = (255, 165, 0)
        elif self.last_prediction in {
            "Waiting...", "NO OBJECT", "DETECTION ERROR", "MODEL ERROR",
        }:
            colour = (210, 210, 210)
        else:
            colour = (0, 255, 0)

        header = (
            f"{MODE_SHORT.get(mode, mode.upper())} | "
            f"frames={len(self.temporal_history)}/"
            f"{max(1, self.temporal_frames)}"
        )

        cv2.putText(
            frame, header, (x1 + 14, y1 + 27), cv2.FONT_HERSHEY_SIMPLEX,
            0.47, colour, 1, cv2.LINE_AA,
        )
        cv2.putText(
            frame, self.last_prediction, (x1 + 14, y1 + 67),
            cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"Confidence: {self.last_confidence * 100:.2f}%",
            (x1 + 14, y1 + 98), cv2.FONT_HERSHEY_SIMPLEX, 0.57,
            (225, 225, 225), 1, cv2.LINE_AA,
        )

        if mode == "roi":
            detection_text = "Detection disabled"
        elif self.last_yolo_error is not None:
            detection_text = "Detection error"
        elif self.last_detection_found:
            detection_text = "Object detected"
        else:
            detection_text = "No object detected"

        cv2.putText(
            frame, f"{detection_text} | FPS: {self.fps:.1f}",
            (x1 + 14, y1 + 124), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (190, 190, 190), 1, cv2.LINE_AA,
        )

    # -- readers -----------------------------------------------------------

    def get_snapshot(self):
        """Return copies of the most recent images, for download."""
        with self.lock:
            return (
                None if self.last_display_rgb is None
                else self.last_display_rgb.copy(),
                None if self.last_classifier_rgb is None
                else self.last_classifier_rgb.copy(),
                None if self.last_input_rgb is None
                else self.last_input_rgb.copy(),
            )

    def get_status(self) -> dict:
        with self.lock:
            return {
                "prediction": self.last_prediction,
                "confidence": self.last_confidence,
                "detected": self.last_detection_found,
                "fps": self.fps,
                "probabilities": (
                    None if self.last_probabilities is None
                    else np.asarray(self.last_probabilities).copy()
                ),
            }

    # -- frame callback ----------------------------------------------------

    def recv(self, frame):
        """Process one frame from the browser camera."""
        rgb_frame = frame.to_ndarray(format="rgb24")
        rgb_frame = np.ascontiguousarray(rgb_frame[:, ::-1])

        settings = self.read_settings()

        if settings["history_maxlen"] is not None:
            self.temporal_history = deque(maxlen=settings["history_maxlen"])
            self.clear_pending_history_flags()
        elif settings["history_reset"]:
            self.temporal_history.clear()
            self.clear_pending_history_flags()

        mode = settings["input_mode"]
        self.frame_index += 1

        roi_box = self.roi_bounds_for(rgb_frame, settings["roi_fractions"])
        roi_x1, roi_y1, roi_x2, roi_y2 = roi_box
        roi_crop = rgb_frame[roi_y1:roi_y2, roi_x1:roi_x2]

        # Detection runs at an interval; the box is reused in between.
        detection_error = None
        if mode in ("roi_with_yolo", "yolo"):
            if self.frame_index % settings["yolo_every_n"] == 1:
                try:
                    self.cached_box = self.detect_best_object(
                        rgb_frame, roi_box, settings["yolo_threshold"]
                    )
                except Exception as error:
                    self.cached_box = None
                    detection_error = str(error)
        else:
            self.cached_box = None

        detected_box = self.cached_box

        if mode in ("roi", "roi_with_yolo"):
            crop = roi_crop
        elif detected_box is not None:
            box_x1, box_y1, box_x2, box_y2 = detected_box
            crop = rgb_frame[box_y1:box_y2, box_x1:box_x2]
        else:
            crop = None

        crop_usable = (
            crop is not None
            and crop.size > 0
            and crop.shape[0] > 20
            and crop.shape[1] > 20
        )

        should_classify = (
            self.frame_index % settings["classify_every_n"] == 1
        )

        result = None
        if should_classify:
            if not crop_usable:
                self.temporal_history.clear()
                result = (
                    "DETECTION ERROR" if detection_error else "NO OBJECT",
                    0.0, None, None, None,
                )
            elif (
                BLANK_FRAME_GUARD
                and self.content_ratio(crop) < MIN_CONTENT_RATIO
            ):
                self.temporal_history.clear()
                result = ("NO OBJECT", 0.0, None, None, None)
            else:
                try:
                    label, confidence, probabilities, model_input = (
                        self.classify(crop)
                    )
                    result = (
                        label, confidence, probabilities, model_input,
                        crop.copy(),
                    )
                except Exception as error:
                    self.temporal_history.clear()
                    result = ("MODEL ERROR", 0.0, None, None, None)
                    print(f"Classification error: {error}")

        with self.lock:
            self.last_yolo_error = detection_error
            self.last_detection_found = detected_box is not None

            if result is not None:
                (
                    self.last_prediction,
                    self.last_confidence,
                    self.last_probabilities,
                    self.last_input_rgb,
                    self.last_classifier_rgb,
                ) = result

            self.fps_counter += 1
            elapsed = time.perf_counter() - self.fps_start_time
            if elapsed >= 1.0:
                self.fps = self.fps_counter / elapsed
                self.fps_counter = 0
                self.fps_start_time = time.perf_counter()

            display_frame = rgb_frame.copy()
            self.draw_rectangle(display_frame, roi_box, (255, 255, 0), 3)
            cv2.putText(
                display_frame,
                ZONE_LABELS.get(mode, "CLASSIFICATION ZONE"),
                (roi_x1, max(25, roi_y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA,
            )

            if detected_box is not None:
                self.draw_rectangle(
                    display_frame, detected_box, (0, 255, 0), 3
                )

            self.draw_panel(display_frame, mode)
            self.last_display_rgb = display_frame.copy()

        # Presentation timestamps must be preserved or playback stutters.
        output_frame = av.VideoFrame.from_ndarray(display_frame, format="rgb24")
        output_frame.pts = frame.pts
        output_frame.time_base = frame.time_base
        return output_frame


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------

camera_column, information_column = st.columns([3, 1], gap="large")

with camera_column:
    st.subheader("Live camera")
    webrtc_context = webrtc_streamer(
        key="plastisort-camera",
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={
            "video": {
                "facingMode": "user",
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15},
            },
            "audio": False,
        },
        video_processor_factory=PlasticVideoProcessor,
        video_html_attrs={
            "style": {"width": "100%", "height": "auto"},
            "controls": False,
            "autoPlay": True,
        },
        # Asynchronous processing drops stale frames rather than blocking the
        # transport thread while inference runs.
        async_processing=True,
    )

processor = (
    webrtc_context.video_processor if webrtc_context is not None else None
)

if processor is not None and roi_is_valid:
    processor.update_settings(
        input_mode,
        temporal_frames,
        yolo_confidence,
        classify_every_n,
        yolo_every_n,
        (roi_left, roi_top, roi_right, roi_bottom),
    )


# --------------------------------------------------------------------------
# Status panel
# --------------------------------------------------------------------------

def render_status_panel() -> None:
    """Render live metrics. Refreshed on a timer where supported."""
    active = (
        webrtc_context.video_processor if webrtc_context is not None else None
    )

    if active is None:
        st.caption("Start the camera to view live results.")
        return

    status = active.get_status()
    st.metric("Prediction", status["prediction"])
    st.metric("Confidence", f"{status['confidence'] * 100:.2f}%")
    st.metric(
        "Detection",
        "Disabled" if input_mode == "roi"
        else ("Object found" if status["detected"] else "No object"),
    )
    st.metric("Frames per second", f"{status['fps']:.1f}")

    if show_probabilities and status["probabilities"] is not None:
        st.caption("Class probabilities")
        st.dataframe(
            {
                "Class": CLASS_NAMES,
                "Probability": [
                    f"{float(p) * 100:.2f}%" for p in status["probabilities"]
                ],
            },
            hide_index=True,
            use_container_width=True,
        )


# st.fragment requires Streamlit 1.37 or later. Without it the metrics update
# only when the page reruns.
if hasattr(st, "fragment"):
    render_status_panel = st.fragment(run_every=1.0)(render_status_panel)

with information_column:
    st.subheader("Pipeline")
    st.markdown(
        """
        1. The camera frame is mirrored.
        2. **ROI only:** the yellow zone is classified directly.
        3. **ROI + YOLO:** the same zone is classified, with a detection box
           drawn for reference.
        4. **YOLO crop:** only the detected object is classified.
        5. Mode changes take effect without restarting the camera.
        """
    )
    render_status_panel()


# --------------------------------------------------------------------------
# Snapshots
# --------------------------------------------------------------------------

for key in ("snapshot_display", "snapshot_crop", "snapshot_input"):
    if key not in st.session_state:
        st.session_state[key] = None


def image_to_png_bytes(rgb_image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb_image).save(buffer, format="PNG")
    return buffer.getvalue()


st.divider()
snapshot_column, help_column = st.columns([1, 3])

with snapshot_column:
    if st.button(
        "Capture current images",
        use_container_width=True,
        disabled=processor is None,
    ):
        display_image, crop_image, input_image = processor.get_snapshot()
        if display_image is not None:
            st.session_state.snapshot_display = image_to_png_bytes(display_image)
        if crop_image is not None:
            st.session_state.snapshot_crop = image_to_png_bytes(crop_image)
        if input_image is not None:
            st.session_state.snapshot_input = image_to_png_bytes(input_image)

with help_column:
    st.caption(
        "Captures the annotated frame, the crop passed to the classifier, and "
        "the resized 224x224 tensor the model received."
    )

download_columns = st.columns(3)

if st.session_state.snapshot_display is not None:
    download_columns[0].download_button(
        "Annotated frame",
        data=st.session_state.snapshot_display,
        file_name="plastisort_frame.png",
        mime="image/png",
        use_container_width=True,
    )

if st.session_state.snapshot_crop is not None:
    download_columns[1].download_button(
        "Classifier crop",
        data=st.session_state.snapshot_crop,
        file_name="plastisort_crop.png",
        mime="image/png",
        use_container_width=True,
    )

if st.session_state.snapshot_input is not None:
    download_columns[2].download_button(
        "Model input",
        data=st.session_state.snapshot_input,
        file_name="plastisort_model_input.png",
        mime="image/png",
        use_container_width=True,
    )

st.divider()
st.caption(
    "Class order: HDPE, LDPE, PET, PP, PS, UNKNOWN. UNKNOWN is a trained "
    "output class; low-confidence predictions are not relabelled."
)
