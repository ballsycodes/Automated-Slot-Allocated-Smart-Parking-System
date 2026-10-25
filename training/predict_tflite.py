import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"C:\Users\druva\final_project\Main_sampling_data\to_esp\parking_cnn_150_int8.tflite"

IMAGE_PATH = r"C:\Users\druva\final_project\Main_sampling_data\CNR-EXT-Patches-150x150\PATCHES\SUNNY\2015-12-10\camera4\S_2015-12-10_11.48_C04_608.jpg"

IMAGE_SIZE = (150, 150)


# ============================================================
# LOAD MODEL
# ============================================================

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print("TFLite model loaded successfully.")
print("Input shape:", input_details["shape"])
print("Input type:", input_details["dtype"])
print("Output shape:", output_details["shape"])
print("Output type:", output_details["dtype"])


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(IMAGE_PATH).convert("RGB")

original_image = image.copy()


# ============================================================
# PREPROCESS AND QUANTIZE
# ============================================================

image_resized = image.resize(IMAGE_SIZE)

image_array = np.array(image_resized).astype("float32") / 255.0

input_image = np.expand_dims(image_array, axis=0)

input_scale, input_zero_point = input_details["quantization"]

if input_scale == 0:
    raise ValueError("The TFLite input tensor has no valid quantization scale.")

quantized_input = np.round(
    input_image / input_scale + input_zero_point
).clip(-128, 127).astype(input_details["dtype"])


# ============================================================
# PREDICT
# ============================================================

interpreter.set_tensor(input_details["index"], quantized_input)
interpreter.invoke()

quantized_output = interpreter.get_tensor(output_details["index"])

output_scale, output_zero_point = output_details["quantization"]

if output_scale == 0:
    raise ValueError("The TFLite output tensor has no valid quantization scale.")

probabilities = (
    quantized_output.astype("float32") - output_zero_point
) * output_scale
probabilities = probabilities[0]

# The model's final layer is softmax. Normalize after dequantization because
# int8 rounding can make the two values sum to slightly less than one.
probability_total = probabilities.sum()

if probability_total <= 0:
    raise ValueError("The TFLite model returned invalid output probabilities.")

probabilities = probabilities / probability_total

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

if np.max(probabilities) >= 0.999:
    print("\nWarning: The model output is saturated. The confidence is genuine, but")
    print("the model is making an almost certain prediction for this image.")


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
