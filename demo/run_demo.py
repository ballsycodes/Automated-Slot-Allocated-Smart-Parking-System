"""Regenerate the metrics and sample outputs for the INT8 parking-slot classifier.

Runs the deployed TFLite model (model/parking_cnn_150_int8.tflite) exactly the
way the ESP32-S3 firmware does: 150x150 RGB input, INT8 input quantization,
INT8 output, dequantization, argmax (0 = EMPTY, 1 = OCCUPIED).

Usage (from the repository root):
    python demo/run_demo.py                          # 60 bundled sample patches
    python demo/run_demo.py --dataset PATH/TO/CNR-EXT-Patches-150x150   # full test split
"""
import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "model" / "parking_cnn_150_int8.tflite"
SAMPLES = ROOT / "demo" / "samples"
RESULTS = ROOT / "demo" / "results"
CLASSES = ["EMPTY", "OCCUPIED"]


def load_interpreter():
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(MODEL))
        runtime = f"tensorflow {tf.__version__}"
    except ImportError:
        from tflite_runtime.interpreter import Interpreter
        interpreter = Interpreter(model_path=str(MODEL))
        runtime = "tflite_runtime"
    interpreter.allocate_tensors()
    return interpreter, runtime


def classify(interpreter, image_path):
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    size = tuple(int(v) for v in inp["shape"][1:3][::-1])

    image = Image.open(image_path).convert("RGB").resize(size)
    x = np.asarray(image, dtype="float32")[None] / 255.0
    scale, zero = inp["quantization"]
    q = np.round(x / scale + zero).clip(-128, 127).astype(inp["dtype"])

    start = time.perf_counter()
    interpreter.set_tensor(inp["index"], q)
    interpreter.invoke()
    elapsed_ms = (time.perf_counter() - start) * 1000

    o_scale, o_zero = out["quantization"]
    probs = (interpreter.get_tensor(out["index"])[0].astype("float32") - o_zero) * o_scale
    probs = probs / probs.sum()
    return int(np.argmax(probs)), probs, elapsed_ms


def read_items(dataset):
    """Return [(image_path, label)] from the bundled samples or the full CNR-EXT test split."""
    if dataset is None:
        with open(SAMPLES / "labels.csv", newline="") as f:
            return [(SAMPLES / r["path"], int(r["label"])) for r in csv.DictReader(f)]
    dataset = Path(dataset)
    items = []
    for line in open(dataset / "LABELS" / "test.txt"):
        line = line.strip()
        if line:
            path, label = line.rsplit(" ", 1)
            items.append((dataset / "PATCHES" / path, int(label)))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="CNR-EXT-Patches-150x150 folder (evaluates the full test split)")
    args = ap.parse_args()

    interpreter, runtime = load_interpreter()
    items = read_items(args.dataset)
    tag = "full_test" if args.dataset else "samples"
    RESULTS.mkdir(parents=True, exist_ok=True)

    rows, times = [], []
    confusion = np.zeros((2, 2), dtype=int)  # [true][pred]
    for n, (path, label) in enumerate(items, 1):
        pred, probs, ms = classify(interpreter, path)
        confusion[label, pred] += 1
        times.append(ms)
        rows.append([path.name, CLASSES[label], CLASSES[pred], f"{probs[0]:.4f}", f"{probs[1]:.4f}", f"{ms:.2f}"])
        if n % 5000 == 0:
            print(f"  {n}/{len(items)}")

    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    tp, fn = int(confusion[1, 1]), int(confusion[1, 0])
    fp, tn = int(confusion[0, 1]), int(confusion[0, 0])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    metrics = {
        "evaluated_on": "CNR-EXT test split (full)" if args.dataset else "60 bundled CNR-EXT test patches (30 empty, 30 occupied)",
        "images": total,
        "accuracy": round(correct / total, 4),
        "precision_occupied": round(precision, 4),
        "recall_occupied": round(recall, 4),
        "f1_occupied": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "confusion_matrix": {"true_empty": {"pred_empty": tn, "pred_occupied": fp},
                             "true_occupied": {"pred_empty": fn, "pred_occupied": tp}},
        "pc_inference_ms_mean": round(float(np.mean(times)), 3),
        "model_file": MODEL.name,
        "model_size_bytes": MODEL.stat().st_size,
        "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "input_shape": [int(v) for v in interpreter.get_input_details()[0]["shape"]],
        "runtime": runtime,
        "python": platform.python_version(),
        "note": "ESP32-S3 on-device inference time (about 497 ms) was measured separately on the board; "
                "the PC time above is not comparable.",
    }
    (RESULTS / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2))
    with open(RESULTS / f"predictions_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "true", "predicted", "p_empty", "p_occupied", "pc_ms"])
        w.writerows(rows[:2000])

    print(json.dumps(metrics, indent=2))

    if not args.dataset:  # sample-output grid (only for the small bundled set)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 6, figsize=(12, 6.6))
        pick = list(range(0, 27, 3)) + list(range(30, 57, 3))  # 9 empty + 9 occupied
        for ax, i in zip(axes.flat, pick):
            path, label = items[i]
            pred, probs, _ = classify(interpreter, path)
            ax.imshow(Image.open(path).convert("RGB"))
            ok = pred == label
            ax.set_title(f"{CLASSES[pred]} {probs[pred] * 100:.0f}%", fontsize=9,
                         color="green" if ok else "red")
            ax.axis("off")
        fig.suptitle("INT8 TFLite predictions on CNR-EXT test patches (green = correct)")
        fig.tight_layout()
        fig.savefig(RESULTS / "sample_predictions.png", dpi=110)


if __name__ == "__main__":
    main()
