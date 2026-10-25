import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"C:\Users\druva\final_project\Main_sampling_data\parking_cnn_150.keras"

IMAGE_PATH = r"C:\Users\druva\final_project\Main_sampling_data\CNR-EXT-Patches-150x150\PATCHES\SUNNY\2015-12-10\camera4\S_2015-12-10_08.18_C04_195.jpg"

IMAGE_SIZE = (150, 150)


# ============================================================
# LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print("Parameters:", model.count_params())


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(IMAGE_PATH).convert("RGB")

original_image = image.copy()


# ============================================================
# PREPROCESS
# ============================================================

image_resized = image.resize(IMAGE_SIZE)

image_array = np.array(image_resized)

image_array = image_array.astype("float32") / 255.0

# Add batch dimension
input_image = np.expand_dims(
    image_array,
    axis=0
)


# ============================================================
# PREDICT
# ============================================================

prediction = model.predict(
    input_image,
    verbose=0
)

probabilities = prediction[0]

predicted_class = np.argmax(probabilities)

confidence = probabilities[predicted_class] * 100


# ============================================================
# CLASS NAME
# ============================================================

if predicted_class == 0:
    class_name = "EMPTY"
else:
    class_name = "OCCUPIED"


# ============================================================
# PRINT RESULT
# ============================================================

print("\n==============================")
print("PREDICTION")
print("==============================")

print("Prediction :", class_name)
print("Confidence :", f"{confidence:.2f}%")

print("\nProbabilities:")
print("EMPTY      :", f"{probabilities[0] * 100:.2f}%")
print("OCCUPIED   :", f"{probabilities[1] * 100:.2f}%")


# ============================================================
# DISPLAY
# ============================================================

plt.figure(figsize=(7, 7))

plt.imshow(original_image)

plt.title(
    f"Prediction: {class_name}\n"
    f"Confidence: {confidence:.2f}%"
)

plt.axis("off")

plt.tight_layout()

plt.show()