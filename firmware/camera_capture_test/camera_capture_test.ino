#include <Arduino.h>
#include "esp_camera.h"
#include "board_config.h"

void setup() {
  Serial.begin(115200);
  delay(3000);

  Serial.println();
  Serial.println("================================");
  Serial.println("ESP32-S3 OV3660 Capture Test");
  Serial.println("================================");

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

  // Raw image format for later ROI processing
  config.pixel_format = PIXFORMAT_RGB565;

  // 320 × 240
  config.frame_size = FRAMESIZE_QVGA;

  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.fb_count = 1;

  Serial.println("Initializing camera...");

  esp_err_t err = esp_camera_init(&config);

  if (err != ESP_OK) {
    Serial.printf("Camera initialization failed: 0x%x\n", err);
    return;
  }

  Serial.println("Camera initialized successfully.");

  sensor_t* sensor = esp_camera_sensor_get();

  if (sensor != nullptr) {
    Serial.printf("Sensor PID: 0x%02X\n", sensor->id.PID);
    Serial.printf("Sensor VER: 0x%02X\n", sensor->id.VER);

    if (sensor->id.PID == OV3660_PID) {
      sensor->set_vflip(sensor, 1);
      sensor->set_brightness(sensor, 1);
      sensor->set_saturation(sensor, -2);
    }
  }

  Serial.printf("Free heap: %u bytes\n", ESP.getFreeHeap());
  Serial.printf("Free PSRAM: %u bytes\n", ESP.getFreePsram());

  Serial.println("Capturing frame...");

  unsigned long startTime = millis();

  camera_fb_t* frame = esp_camera_fb_get();

  unsigned long captureTime = millis() - startTime;

  if (frame == nullptr) {
    Serial.println("Frame capture failed.");
    return;
  }

  Serial.println("Frame captured successfully.");

  Serial.printf("Width: %u pixels\n", frame->width);
  Serial.printf("Height: %u pixels\n", frame->height);
  Serial.printf("Length: %u bytes\n", frame->len);
  Serial.printf("Format ID: %d\n", frame->format);
  Serial.printf("Capture time: %lu ms\n", captureTime);

  Serial.printf(
    "Expected RGB565 size: %u bytes\n",
    frame->width * frame->height * 2
  );

  Serial.print("First 16 bytes: ");

  for (int i = 0; i < 16 && i < frame->len; i++) {
    Serial.printf("%02X ", frame->buf[i]);
  }

  Serial.println();

  esp_camera_fb_return(frame);

  Serial.printf(
    "Free PSRAM after returning frame: %u bytes\n",
    ESP.getFreePsram()
  );

  Serial.println("Camera test completed.");
}

void loop() {
  delay(5000);
}