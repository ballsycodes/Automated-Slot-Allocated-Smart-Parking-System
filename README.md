# Smart Parking Slot Detection with TinyML on ESP32-S3

Team 79 - B.Tech CSE, Amrita Vishwa Vidyapeetham (Coimbatore). Guide: Dr. R. Karthi.
Project: *Automated Slot-Allocated Smart Parking System* (Application project, Project Phase II).

A camera-equipped **ESP32-S3** classifies a parking region as **EMPTY** or **OCCUPIED** fully on-device, using a 25,730-parameter CNN quantized to INT8 (33 KB) and run with TensorFlow Lite Micro. Results are shown on a local web dashboard served by the board itself.

## Status (honest summary)

| Part | State |
|---|---|
| CNN training on CNR-EXT, INT8 TFLite conversion | Done |
| ESP32-S3 firmware: camera capture, preprocessing, on-device inference | Done |
| ESP32 Wi-Fi access-point dashboard (`192.168.4.1`) | Done (works on laptop USB power; power-bank boot instability still being debugged) |
| Per-slot ROI mapping, retraining on campus images | Planned |
| LoRa link, Raspberry Pi backend (Flask/SQLite), QR entry, SMS | Planned |
| Priority/congestion-aware slot allocation, time-window wrong-parking detection | Designed, not implemented |

Known limitations:
- Rotation sensitivity: a synthetic 5 degree rotation drops INT8 accuracy to about 79% (partly an artefact of black corners, not yet verified on real tilted captures).
- The model was trained on CNR-EXT patches of single slots; the current demo feeds the whole frame resized to 150x150, so people or other objects can be classified as occupied. Per-slot ROI crops and campus data are the next step.

## Results

| Metric | Value | Source |
|---|---|---|
| Test accuracy, full CNR-EXT test split (31,825 images), INT8 TFLite | 98.01% | `python demo/run_demo.py --dataset ...` |
| Keras float model test accuracy (150x150) | 98.07% re-measured (98.11% in the training log) | `python demo/analysis.py --dataset ...` |
| Float vs INT8 agreement | 99.66% of patches | `demo/results/analysis/analysis_summary.json` |
| Accuracy by weather (INT8) | Overcast 99.14%, Sunny 98.34%, Rainy 97.14% | same file |
| Camera-shift test (4,000 test patches, INT8) | shifts/zoom/brightness: 96.3-98.1%; **5 degree rotation: 79.1%** | same file (synthetic, black corners on rotation) |
| Parameters | 25,730 | model summary |
| Model size | FP32 about 100.5 KB, INT8 TFLite 33.24 KB (34,040 B) | `model/` |
| ESP32-S3 inference time | about 497 ms (first capture about 300 ms, later about 7 ms) | measured on board, serial log |
| Input | 150 x 150 x 3, INT8 | |

These numbers are CNR-EXT benchmark results, not results on a deployed campus lot.

## Repository layout

```
firmware/    Arduino sketches for ESP32-S3 (OV3660)
  camera_capture_test/     camera init + one frame (320x240 RGB565)
  parking_tflite_test/     TFLite Micro with a synthetic input
  camera_tflite_test/      camera frame -> CNN -> serial output
  smart_parking_ap_demo/   capture + infer every 30 s, Wi-Fi AP dashboard
model/       parking_cnn_150_int8.tflite, Keras models, C array (model_data.cc/.h)
training/    dataset check/preprocess, training, TFLite conversion, PC prediction
demo/        run_demo.py, sample patches, results, demo videos  (see demo/README.md)
docs/        project website (GitHub Pages)
paper/       research paper draft: template.html, build_paper.py, make_figures.py, paper.pdf
```

## Quick start - reproduce the metrics (no hardware needed)

```bash
pip install -r requirements.txt
python demo/run_demo.py                 # 60 bundled test patches -> demo/results/
python demo/run_demo.py --dataset /path/to/CNR-EXT-Patches-150x150   # full test split
```

Outputs: `demo/results/metrics_*.json`, `predictions_*.csv`, `sample_predictions.png`. Details in [demo/README.md](demo/README.md).

## Hardware demo (ESP32-S3)

Board used: ESP32-S3-WROOM-1 N16R8 with OV3660 camera (8 MB PSRAM, 16 MB flash).

1. Arduino IDE with the ESP32 board package; board **ESP32S3 Dev Module**, **USB CDC On Boot: Enabled**, **PSRAM: OPI PSRAM**.
2. Install the Espressif TensorFlow Lite Micro Arduino library (`esp-tflite-micro`).
3. Open `firmware/smart_parking_ap_demo/smart_parking_ap_demo.ino` and upload.
4. Join Wi-Fi `SmartParking-ESP32-1` (demo-only password is set in the sketch - change it), browse to `http://192.168.4.1`.
5. The dashboard shows the latest image, prediction, confidence, capture time and inference time, refreshed every 30 s.

Pipeline on the board: RGB565 frame -> resize 150x150 -> RGB888 -> INT8 input -> TFLite Micro -> INT8 output -> dequantize -> EMPTY/OCCUPIED. The tensor arena is allocated in PSRAM.

## Training

Dataset: CNRPark+EXT patches (150x150) from the original authors' site (not included; very large). The scripts in `training/` use absolute paths at the top - edit them for your machine.

```
check_dataset.py  ->  preprocess_dataset.py  ->  train_baseline.py (150x150)  ->  Convert_150_to_tflite.py  ->  convert_model_to_c.py
```

Architecture: Conv16 - MaxPool - Conv32 - MaxPool - Conv64 - GlobalAveragePooling - Dense32 - Dropout(0.3) - Dense2 (softmax). Trained on a balanced 20,000-image subset for up to 20 epochs with light augmentation.

## Design (planned system)

ESP32-S3 camera nodes (TinyML occupancy) -> LoRa -> Raspberry Pi 3A+ (Flask, SQLite, dynamic QR entry, allocation engine, SMS) -> Supabase cloud sync. Allocation score per free slot: `w1*D_entrance + w2*D_lift + w3*(row_occupancy + alpha*recent_allocations)`, lowest score wins. Wrong parking is detected by a time-window check between assigned and newly occupied slots.

## Team

| Name | Register No. | Contribution |
|---|---|---|
| Druva Teja | CB.SC.U4CSE23317 | CNN model, quantization, TinyML pipeline |
| Sai Jeevan | CB.SC.U4CSE23124 | ESP32-S3 hardware, camera, dashboard |
| Manish Sakamuri | CB.SC.U4CSE23125 | System architecture, IoT integration |
| Tejeswara D | CB.SC.U4CSE23614 | Allocation algorithm, backend design |

## Credits and license

Code: MIT (see `LICENSE`). Dataset: M. Amato et al., CNRPark+EXT; sample patches included for demonstration only under the dataset's terms.
