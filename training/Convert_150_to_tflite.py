import tensorflow as tf
import numpy as np
import os

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"C:\Users\druva\final_project\Main_sampling_data\parking_cnn_150.keras"

OUTPUT_PATH = r"C:\Users\druva\final_project\Main_sampling_data\to_esp\parking_cnn_150_int8.tflite"

# Change this to your actual CNR-EXT PATCHES directory
DATASET_DIR = r"C:\Users\druva\final_project\Main_sampling_data\CNR-EXT-Patches-150x150\PATCHES"

IMAGE_SIZE = (150, 150)

# Number of images used for calibration
CALIBRATION_IMAGES = 500


# ============================================================
# LOAD KERAS MODEL
# ============================================================

print("Loading Keras model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded.")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Parameters:", model.count_params())


# ============================================================
# FIND CALIBRATION IMAGES
# ============================================================

image_paths = []

for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(root, file))

if len(image_paths) == 0:
    raise RuntimeError(
        "No images found. Check DATASET_DIR."
    )

np.random.seed(42)
np.random.shuffle(image_paths)

image_paths = image_paths[:CALIBRATION_IMAGES]

print("Calibration images:", len(image_paths))


# ============================================================
# REPRESENTATIVE DATASET
# ============================================================

def representative_dataset():
    for image_path in image_paths:
        image = tf.keras.utils.load_img(
            image_path,
            target_size=IMAGE_SIZE,
            color_mode="rgb"
        )

        image_array = tf.keras.utils.img_to_array(image)

        # Same preprocessing used during training
        image_array = image_array.astype(np.float32) / 255.0

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        yield [image_array]


# ============================================================
# CONVERT TO TFLITE
# ============================================================

print("Converting to INT8 TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.representative_dataset = representative_dataset

# Force integer operations
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

# Input and output will also be INT8
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()


# ============================================================
# SAVE MODEL
# ============================================================

with open(OUTPUT_PATH, "wb") as file:
    file.write(tflite_model)

print("\n================================")
print("CONVERSION COMPLETE")
print("================================")

print("Output:", OUTPUT_PATH)
print("TFLite size:", len(tflite_model), "bytes")
print("TFLite size:", round(len(tflite_model) / 1024, 2), "KB")