#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include "esp_camera.h"

#include "board_config.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

#define MODEL_WIDTH 150
#define MODEL_HEIGHT 150
#define CAPTURE_INTERVAL 30000UL
constexpr size_t TENSOR_ARENA_SIZE = 600 * 1024;

const char* AP_SSID = "SmartParking-ESP32-1";
const char* AP_PASSWORD = "12345678";

WebServer server(80);
uint8_t* tensor_arena = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

camera_fb_t* saved_frame = nullptr;
String prediction = "Not captured";
float empty_probability = 0;
float occupied_probability = 0;
unsigned long capture_time = 0;
unsigned long inference_time = 0;
unsigned long last_capture = 0;
bool ready = false;

void rgb565ToRgb888(uint16_t p, uint8_t& r, uint8_t& g, uint8_t& b) {
  r = ((p >> 11) & 0x1F) * 255 / 31;
  g = ((p >> 5) & 0x3F) * 255 / 63;
  b = (p & 0x1F) * 255 / 31;
}

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
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size = FRAMESIZE_QVGA;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.fb_count = 1;

  if (esp_camera_init(&config) != ESP_OK) return false;

  sensor_t* sensor = esp_camera_sensor_get();
  if (sensor && sensor->id.PID == OV3660_PID) {
    sensor->set_vflip(sensor, 1);
    sensor->set_brightness(sensor, 1);
    sensor->set_saturation(sensor, -2);
  }
  return true;
}

bool initializeModel() {
  model = tflite::GetModel(parking_cnn_150_int8_tflite);
  if (!model || model->version() != TFLITE_SCHEMA_VERSION) return false;

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
  if (!tensor_arena) return false;

  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, TENSOR_ARENA_SIZE
  );
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) return false;

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);
  return true;
}

void prepareModelInput(camera_fb_t* frame) {
  for (int y = 0; y < MODEL_HEIGHT; y++) {
    int sy = y * frame->height / MODEL_HEIGHT;

    for (int x = 0; x < MODEL_WIDTH; x++) {
      int sx = x * frame->width / MODEL_WIDTH;
      int source_index = (sy * frame->width + sx) * 2;

      uint16_t pixel =
        ((uint16_t)frame->buf[source_index] << 8) |
        frame->buf[source_index + 1];

      uint8_t r, g, b;
      rgb565ToRgb888(pixel, r, g, b);

      int output_index = (y * MODEL_WIDTH + x) * 3;
      input_tensor->data.int8[output_index] = (int8_t)((int)r - 128);
      input_tensor->data.int8[output_index + 1] = (int8_t)((int)g - 128);
      input_tensor->data.int8[output_index + 2] = (int8_t)((int)b - 128);
    }
  }
}

void releaseSavedFrame() {
  if (saved_frame) {
    free(saved_frame->buf);
    free(saved_frame);
    saved_frame = nullptr;
  }
}

void captureAndInfer() {
  unsigned long start = millis();
  camera_fb_t* frame = esp_camera_fb_get();

  if (!frame) {
    prediction = "Capture failed";
    return;
  }

  releaseSavedFrame();
  saved_frame = (camera_fb_t*)malloc(sizeof(camera_fb_t));

  if (!saved_frame) {
    esp_camera_fb_return(frame);
    prediction = "Memory allocation failed";
    return;
  }

  memcpy(saved_frame, frame, sizeof(camera_fb_t));
  saved_frame->buf = (uint8_t*)ps_malloc(frame->len);

  if (!saved_frame->buf) {
    releaseSavedFrame();
    esp_camera_fb_return(frame);
    prediction = "PSRAM allocation failed";
    return;
  }

  memcpy(saved_frame->buf, frame->buf, frame->len);
  esp_camera_fb_return(frame);

  capture_time = millis() - start;
  prepareModelInput(saved_frame);

  start = millis();
  if (interpreter->Invoke() != kTfLiteOk) {
    prediction = "Inference failed";
    inference_time = millis() - start;
    return;
  }

  inference_time = millis() - start;

  float scale = output_tensor->params.scale;
  int zero_point = output_tensor->params.zero_point;

  empty_probability =
    (output_tensor->data.int8[0] - zero_point) * scale;
  occupied_probability =
    (output_tensor->data.int8[1] - zero_point) * scale;

  prediction = empty_probability >= occupied_probability
                 ? "EMPTY"
                 : "OCCUPIED";

  Serial.printf(
    "Prediction: %s | Empty: %.2f%% | Occupied: %.2f%% | Capture: %lu ms | Inference: %lu ms\n",
    prediction.c_str(),
    empty_probability * 100.0f,
    occupied_probability * 100.0f,
    capture_time,
    inference_time
  );
}

void sendImage() {
  if (!saved_frame) {
    server.send(404, "text/plain", "No image available");
    return;
  }

  const int width = saved_frame->width;
  const int height = saved_frame->height;
  const uint32_t row_size = width * 3;
  const uint32_t image_size = row_size * height;
  const uint32_t file_size = 54 + image_size;

  uint8_t header[54] = {};
  header[0] = 'B';
  header[1] = 'M';
  memcpy(header + 2, &file_size, 4);

  uint32_t offset = 54;
  uint32_t dib_size = 40;
  int32_t negative_height = -height;
  uint16_t planes = 1;
  uint16_t bits = 24;

  memcpy(header + 10, &offset, 4);
  memcpy(header + 14, &dib_size, 4);
  memcpy(header + 18, &width, 4);
  memcpy(header + 22, &negative_height, 4);
  memcpy(header + 26, &planes, 2);
  memcpy(header + 28, &bits, 2);

  WiFiClient client = server.client();
  client.printf(
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: image/bmp\r\n"
    "Content-Length: %u\r\n"
    "Connection: close\r\n\r\n",
    file_size
  );
  client.write(header, 54);

  uint8_t row[320 * 3];

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int source_index = (y * width + x) * 2;
      uint16_t pixel =
        ((uint16_t)saved_frame->buf[source_index] << 8) |
        saved_frame->buf[source_index + 1];

      uint8_t r, g, b;
      rgb565ToRgb888(pixel, r, g, b);

      row[x * 3] = b;
      row[x * 3 + 1] = g;
      row[x * 3 + 2] = r;
    }
    client.write(row, row_size);
  }
}

void sendDashboard() {
  String html = R"HTML(
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="5">
<title>Smart Parking</title>
<style>
body{font-family:Arial;background:#f2f2f2;margin:20px}
.card{background:white;padding:20px;border-radius:12px;max-width:600px}
img{width:100%;max-width:320px}
.result{font-size:28px;font-weight:bold}
</style>
</head>
<body>
<div class="card">
<h2>Smart Parking Demo</h2>
<img src="/image">
<p class="result">Prediction: )HTML";

  html += prediction;
  html += R"HTML(</p>
<p>Empty confidence: )HTML";
  html += String(empty_probability * 100.0f, 2);
  html += R"HTML(%</p>
<p>Occupied confidence: )HTML";
  html += String(occupied_probability * 100.0f, 2);
  html += R"HTML(%</p>
<p>Captured image: 320 x 240 RGB565</p>
<p>Model input: 150 x 150 x 3 INT8</p>
<p>Capture time: )HTML";
  html += String(capture_time);
  html += R"HTML( ms</p>
<p>Inference time: )HTML";
  html += String(inference_time);
  html += R"HTML( ms</p>
<p>Free heap: )HTML";
  html += String(ESP.getFreeHeap());
  html += R"HTML( bytes</p>
<p>Free PSRAM: )HTML";
  html += String(ESP.getFreePsram());
  html += R"HTML( bytes</p>
<p>New capture every 30 seconds.</p>
</div>
</body>
</html>
)HTML";

  server.send(200, "text/html", html);
}

void setup() {
  Serial.begin(115200);
  delay(3000);

  Serial.println("\nSmart Parking Camera + TinyML Demo");

  if (!psramFound()) {
    Serial.println("PSRAM not detected");
    return;
  }

  if (!initializeCamera()) {
    Serial.println("Camera initialization failed");
    return;
  }

  if (!initializeModel()) {
    Serial.println("Model initialization failed");
    return;
  }

  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASSWORD);

  Serial.print("Connect to Wi-Fi: ");
  Serial.println(AP_SSID);
  Serial.print("Open: http://");
  Serial.println(WiFi.softAPIP());

  server.on("/", HTTP_GET, sendDashboard);
  server.on("/image", HTTP_GET, sendImage);
  server.begin();

  ready = true;
  captureAndInfer();
  last_capture = millis();
}

void loop() {
  if (!ready) {
    delay(1000);
    return;
  }

  server.handleClient();

  if (millis() - last_capture >= CAPTURE_INTERVAL) {
    captureAndInfer();
    last_capture = millis();
  }

  delay(2);
}
