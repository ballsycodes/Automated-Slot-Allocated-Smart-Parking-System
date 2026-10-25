from pathlib import Path

model_path = Path(
    r"C:\Users\druva\final_project\Main_sampling_data\to_esp\parking_cnn_150_int8.tflite"
)

data = model_path.read_bytes()

with open("model_data.cc", "w") as f:
    f.write("#include <stdint.h>\n\n")
    f.write("const unsigned char parking_cnn_150_int8_tflite[] = {\n")

    for i in range(0, len(data), 12):
        chunk = data[i:i + 12]
        f.write("    " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\n")

    f.write("};\n\n")
    f.write(
        "const unsigned int parking_cnn_150_int8_tflite_len = "
        "sizeof(parking_cnn_150_int8_tflite);\n"
    )