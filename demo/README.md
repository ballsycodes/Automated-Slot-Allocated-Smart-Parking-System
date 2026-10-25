# Working demonstration

Two parts: the hardware prototype (video evidence) and a run-based demo that regenerates the metrics and sample outputs.

## 1. Hardware prototype

ESP32-S3 + OV3660 captures a frame every 30 s, classifies it on-device (INT8 CNN, about 497 ms) and serves the result on its own Wi-Fi access point.

Videos in `media/` (also embedded on the project website):
- `Occupied_LOT.mp4` - prototype pointed at an occupied lot
- `Empty_LOT.mp4` - prototype pointed at an empty lot

Photos from the campus lot used during testing: `media/campus_photos/`.

## 2. Run-based demo (regenerates metrics + sample outputs)

Versions: model `model/parking_cnn_150_int8.tflite` (SHA-256 in `results/metrics_samples.json`), Python 3.13, TensorFlow 2.20 (or `tflite-runtime`).

```bash
pip install -r requirements.txt
python demo/run_demo.py
```

Writes to `demo/results/`:
- `metrics_samples.json` - accuracy, precision, recall, F1, confusion matrix, model size/hash, runtime versions
- `predictions_samples.csv` - per-image prediction and probabilities
- `sample_predictions.png` - grid of sample outputs (green = correct)

The 60 images in `samples/` (30 empty, 30 occupied) are randomly drawn (seed 42) from the CNR-EXT **test** split. 60 images is a smoke test, not a benchmark. For the full test split:

```bash
python demo/run_demo.py --dataset /path/to/CNR-EXT-Patches-150x150
```

which writes `metrics_full_test.json` (a committed copy is included: 98.01% accuracy, 31,825 images).

## Limits

- Numbers are on CNR-EXT patches, not on the deployed campus camera.
- PC inference time in the JSON is not the ESP32 time.
- Whole-frame input on the board can misclassify people/objects; ROI cropping is planned.
