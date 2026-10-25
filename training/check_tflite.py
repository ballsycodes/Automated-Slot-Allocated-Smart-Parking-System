import tensorflow as tf

MODEL_PATH = r"C:\Users\druva\final_project\Main_sampling_data\to_esp\parking_cnn_150_int8.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nINPUT")
print("Shape:", input_details[0]["shape"])
print("Type:", input_details[0]["dtype"])
print("Quantization:", input_details[0]["quantization"])

print("\nOUTPUT")
print("Shape:", output_details[0]["shape"])
print("Type:", output_details[0]["dtype"])
print("Quantization:", output_details[0]["quantization"])