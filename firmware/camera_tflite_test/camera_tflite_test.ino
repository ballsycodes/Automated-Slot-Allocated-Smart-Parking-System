#include <Arduino.h>
#include "esp_camera.h"

#include "board_config.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

// --------------------------------------------------
// Camera settings
// --------------------------------------------------

#define CAMERA_WIDTH  320
#define CAMERA_HEIGHT 240

// --------------------------------------------------
// Model settings
// --------------------------------------------------

#define MODEL_WIDTH  150
#define MODEL_HEIGHT 150
#define MODEL_CHANNELS 3

// Your INT8 model requires approximately 469 KB.
// Use PSRAM for the tensor arena.
constexpr size_t TENSOR_ARENA_SIZE = 600 * 1024;

uint8_t* tensor_arena = nullptr;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// --------------------------------------------------
// Camera initialization
// --------------------------------------------------

bool initializeCamera() {
  camera_config_t config;

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;

  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;

  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;

  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;

  // Raw pixels for local processing
  config.pixel_format = PIXFORMAT_RGB565;

  // 320 × 240
  config.frame_size = FRAMESIZE_QVGA;

  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.fb_count = 1;

  Serial.println("Initializing camera...");

  esp_err_t error = esp_camera_init(&config);

  if (error != ESP_OK) {
    Serial.printf(
      "Camera initialization failed: 0x%x\n",
      error
    );
    return false;
  }

  sensor_t* sensor = esp_camera_sensor_get();

  if (sensor != nullptr) {
    Serial.printf(
      "Camera sensor PID: 0x%02X\n",
      sensor->id.PID
    );

    if (sensor->id.PID == OV3660_PID) {
      sensor->set_vflip(sensor, 1);
      sensor->set_brightness(sensor, 1);
      sensor->set_saturation(sensor, -2);
    }
  }

  Serial.println("Camera initialized successfully.");

  return true;
}

// --------------------------------------------------
// TensorFlow Lite initialization
// --------------------------------------------------

bool initializeModel() {
  Serial.println("Initializing TensorFlow Lite model...");

  model = tflite::GetModel(
    parking_cnn_150_int8_tflite
  );

  if (model == nullptr) {
    Serial.println("Model pointer is null.");
    return false;
  }

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf(
      "Model schema mismatch. Model: %d, Supported: %d\n",
      model->version(),
      TFLITE_SCHEMA_VERSION
    );
    return false;
  }

  // Register the operators used by your CNN.
  static tflite::MicroMutableOpResolver<10> resolver;

  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();

  tensor_arena = (uint8_t*)ps_malloc(TENSOR_ARENA_SIZE);

  if (tensor_arena == nullptr) {
    Serial.println("Tensor arena allocation failed.");
    return false;
  }

  Serial.printf(
    "Tensor arena allocated: %u bytes\n",
    TENSOR_ARENA_SIZE
  );

  static tflite::MicroInterpreter static_interpreter(
    model,
    resolver,
    tensor_arena,
    TENSOR_ARENA_SIZE
  );

  interpreter = &static_interpreter;

  TfLiteStatus status = interpreter->AllocateTensors();

  if (status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed.");
    return false;
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.println("Model initialized successfully.");

  Serial.printf(
    "Input dimensions: %d × %d × %d\n",
    input_tensor->dims->data[1],
    input_tensor->dims->data[2],
    input_tensor->dims->data[3]
  );

  Serial.printf(
    "Input type: %d\n",
    input_tensor->type
  );

  Serial.printf(
    "Input scale: %f\n",
    input_tensor->params.scale
  );

  Serial.printf(
    "Input zero point: %d\n",
    input_tensor->params.zero_point
  );

  Serial.printf(
    "Output scale: %f\n",
    output_tensor->params.scale
  );

  Serial.printf(
    "Output zero point: %d\n",
    output_tensor->params.zero_point
  );

  return true;
}

// --------------------------------------------------
// Convert RGB565 pixel to RGB888
// --------------------------------------------------

void rgb565ToRgb888(
  uint16_t pixel,
  uint8_t& red,
  uint8_t& green,
  uint8_t& blue
) {
  red = ((pixel >> 11) & 0x1F) * 255 / 31;
  green = ((pixel >> 5) & 0x3F) * 255 / 63;
  blue = (pixel & 0x1F) * 255 / 31;
}

// --------------------------------------------------
// Resize complete camera frame to model input
// Nearest-neighbor resizing
// --------------------------------------------------

void prepareModelInput(camera_fb_t* frame) {
  const uint8_t* source = frame->buf;

  for (int output_y = 0; output_y < MODEL_HEIGHT; output_y++) {
    int source_y =
      output_y * frame->height / MODEL_HEIGHT;

    for (int output_x = 0; output_x < MODEL_WIDTH; output_x++) {
      int source_x =
        output_x * frame->width / MODEL_WIDTH;

      int source_index =
        (source_y * frame->width + source_x) * 2;

      uint16_t rgb565 =
        ((uint16_t)source[source_index] << 8) |
        source[source_index + 1];

      uint8_t red;
      uint8_t green;
      uint8_t blue;

      rgb565ToRgb888(
        rgb565,
        red,
        green,
        blue
      );

      int output_index =
        (output_y * MODEL_WIDTH + output_x) * 3;

      // The model quantization is:
      //
      // real_value = scale × (int8_value - zero_point)
      //
      // For this model:
      // scale = 1/255
      // zero_point = -128
      //
      // Therefore:
      // RGB 0   → int8 -128
      // RGB 255 → int8 127

      input_tensor->data.int8[output_index] =
        (int8_t)((int)red - 128);

      input_tensor->data.int8[output_index + 1] =
        (int8_t)((int)green - 128);

      input_tensor->data.int8[output_index + 2] =
        (int8_t)((int)blue - 128);
    }
  }
}

// --------------------------------------------------
// Run inference
// --------------------------------------------------

void runInference(camera_fb_t* frame) {
  prepareModelInput(frame);

  unsigned long startTime = millis();

  TfLiteStatus status = interpreter->Invoke();

  unsigned long inferenceTime =
    millis() - startTime;

  if (status != kTfLiteOk) {
    Serial.println("Inference failed.");
    return;
  }

  int8_t empty_raw =
    output_tensor->data.int8[0];

  int8_t occupied_raw =
    output_tensor->data.int8[1];

  float output_scale =
    output_tensor->params.scale;

  int output_zero_point =
    output_tensor->params.zero_point;

  float empty_probability =
    (empty_raw - output_zero_point) * output_scale;

  float occupied_probability =
    (occupied_raw - output_zero_point) * output_scale;

  Serial.println();
  Serial.println("---------- Inference ----------");

  Serial.printf(
    "Raw EMPTY output: %d\n",
    empty_raw
  );

  Serial.printf(
    "Raw OCCUPIED output: %d\n",
    occupied_raw
  );

  Serial.printf(
    "EMPTY probability: %.4f\n",
    empty_probability
  );

  Serial.printf(
    "OCCUPIED probability: %.4f\n",
    occupied_probability
  );

  if (empty_probability > occupied_probability) {
    Serial.println("Prediction: EMPTY");
  } else {
    Serial.println("Prediction: OCCUPIED");
  }

  Serial.printf(
    "Inference time: %lu ms\n",
    inferenceTime
  );

  Serial.println("-------------------------------");
}

// --------------------------------------------------
// Setup
// --------------------------------------------------

void setup() {
  Serial.begin(115200);
  delay(3000);

  Serial.println();
  Serial.println("================================");
  Serial.println("ESP32-S3 Camera + TinyML Test");
  Serial.println("================================");

  Serial.printf(
    "Free heap before setup: %u bytes\n",
    ESP.getFreeHeap()
  );

  Serial.printf(
    "Free PSRAM before setup: %u bytes\n",
    ESP.getFreePsram()
  );

  if (!psramFound()) {
    Serial.println("PSRAM not detected.");
    return;
  }

  if (!initializeCamera()) {
    return;
  }

  if (!initializeModel()) {
    return;
  }

  Serial.println();
  Serial.println("Camera and model are ready.");
  Serial.println("Capturing real camera frame...");

  camera_fb_t* frame = esp_camera_fb_get();

  if (frame == nullptr) {
    Serial.println("Camera frame capture failed.");
    return;
  }

  Serial.printf(
    "Captured frame: %u × %u\n",
    frame->width,
    frame->height
  );

  Serial.printf(
    "Frame length: %u bytes\n",
    frame->len
  );

  runInference(frame);

  esp_camera_fb_return(frame);

  Serial.printf(
    "Free heap after test: %u bytes\n",
    ESP.getFreeHeap()
  );

  Serial.printf(
    "Free PSRAM after test: %u bytes\n",
    ESP.getFreePsram()
  );

  Serial.println();
  Serial.println("Camera + TinyML test completed.");
}

void loop() {
  delay(5000);
}