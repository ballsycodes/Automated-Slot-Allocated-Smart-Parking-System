#include <Arduino.h>

#include "model_data.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Loaded TensorFlow Lite model
const tflite::Model* model = nullptr;

// TensorFlow Lite interpreter
tflite::MicroInterpreter* interpreter = nullptr;

// Model input and output tensors
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor arena size
// The model previously required approximately 469 KB.
// 600 KB provides some additional space.
constexpr size_t kTensorArenaSize = 600 * 1024;

// Tensor arena will be allocated in external PSRAM
uint8_t* tensor_arena = nullptr;

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println();
  Serial.println("ESP32-S3 TensorFlow Lite Micro Model Test");
  Serial.println("-----------------------------------------");

  // Check whether PSRAM is available
  if (!psramFound()) {
    Serial.println("ERROR: PSRAM not found");
    while (true) {
      delay(1000);
    }
  }

  Serial.print("PSRAM found. Free PSRAM before allocation: ");
  Serial.println(ESP.getFreePsram());

  // Allocate tensor arena in PSRAM
  tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);

  if (tensor_arena == nullptr) {
    Serial.println("ERROR: Failed to allocate tensor arena in PSRAM");
    while (true) {
      delay(1000);
    }
  }

  Serial.print("Tensor arena allocated in PSRAM: ");
  Serial.print(kTensorArenaSize);
  Serial.println(" bytes");

  Serial.print("Free PSRAM after allocation: ");
  Serial.println(ESP.getFreePsram());

  // Load the model
  model = tflite::GetModel(
    parking_cnn_150_int8_tflite
  );

  if (model == nullptr) {
    Serial.println("ERROR: Model could not be loaded");
    return;
  }

  Serial.println("Model loaded successfully");

  // Check the model schema version
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: Model schema version is ");
    Serial.print(model->version());
    Serial.print(", expected ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    return;
  }

  Serial.println("Model schema version is compatible");

  // Register the operations used by the CNN
  static tflite::MicroMutableOpResolver<8> resolver;

  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();

  Serial.println("Operators registered successfully");

  // Create the interpreter
  // static is used so the interpreter remains alive after setup().
  // This is different from the tensor arena, which is allocated in PSRAM.
  static tflite::MicroInterpreter static_interpreter(
    model,
    resolver,
    tensor_arena,
    kTensorArenaSize,
    nullptr
  );

  interpreter = &static_interpreter;

  Serial.println("Interpreter created successfully");

  // Allocate memory for the model
  TfLiteStatus status = interpreter->AllocateTensors();

  if (status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed");
    Serial.println("Try increasing kTensorArenaSize to 768 * 1024");
    return;
  }

  Serial.println("Tensor arena allocated successfully");

  // Get the model input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  if (input == nullptr || output == nullptr) {
    Serial.println("ERROR: Input or output tensor is null");
    return;
  }

  // Display input information
  Serial.println();
  Serial.println("INPUT INFORMATION");

  Serial.print("Type: ");
  Serial.println(input->type);

  Serial.print("Shape: ");

  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);

    if (i < input->dims->size - 1) {
      Serial.print(" x ");
    }
  }

  Serial.println();

  Serial.print("Input scale: ");
  Serial.println(input->params.scale, 8);

  Serial.print("Input zero point: ");
  Serial.println(input->params.zero_point);

  // Display output information
  Serial.println();
  Serial.println("OUTPUT INFORMATION");

  Serial.print("Type: ");
  Serial.println(output->type);

  Serial.print("Shape: ");

  for (int i = 0; i < output->dims->size; i++) {
    Serial.print(output->dims->data[i]);

    if (i < output->dims->size - 1) {
      Serial.print(" x ");
    }
  }

  Serial.println();

  Serial.print("Output scale: ");
  Serial.println(output->params.scale, 8);

  Serial.print("Output zero point: ");
  Serial.println(output->params.zero_point);

  // Fill the input with a neutral gray image.
  //
  // The model input is:
  // 150 x 150 x 3
  //
  // Input quantization:
  // scale = 0.003921568859
  // zero point = -128
  //
  // Pixel value 128 becomes INT8 value 0.
  const int input_length = 150 * 150 * 3;

  for (int i = 0; i < input_length; i++) {
    input->data.int8[i] = 0;
  }

  Serial.println();
  Serial.println("Running test inference...");

  // Execute the model
  status = interpreter->Invoke();

  if (status != kTfLiteOk) {
    Serial.println("ERROR: Inference failed");
    return;
  }

  Serial.println("Inference completed successfully");

  // Read the two output classes
  int8_t empty_raw = output->data.int8[0];
  int8_t occupied_raw = output->data.int8[1];

  // Convert quantized output values to probabilities
  float empty_probability =
    (empty_raw - output->params.zero_point) *
    output->params.scale;

  float occupied_probability =
    (occupied_raw - output->params.zero_point) *
    output->params.scale;

  Serial.println();

  Serial.print("Raw EMPTY output: ");
  Serial.println(empty_raw);

  Serial.print("Raw OCCUPIED output: ");
  Serial.println(occupied_raw);

  Serial.print("EMPTY probability: ");
  Serial.println(empty_probability, 4);

  Serial.print("OCCUPIED probability: ");
  Serial.println(occupied_probability, 4);

  if (empty_probability > occupied_probability) {
    Serial.println("Prediction: EMPTY");
  } else {
    Serial.println("Prediction: OCCUPIED");
  }

  Serial.println();
  Serial.println("Model test completed.");
}

void loop() {
  // Run only once for this test.
}