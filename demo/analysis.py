"""Deeper evaluation used in the paper: float vs INT8, per-weather, per-camera, camera-shift robustness.

    python demo/analysis.py --dataset /path/to/CNR-EXT-Patches-150x150

Writes demo/results/analysis/*.json and the figures used in the manuscript.
Needs tensorflow (for the Keras float model) and matplotlib.
"""
import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "demo" / "results" / "analysis"
SEED = 42


def load_int8():
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=str(ROOT / "model" / "parking_cnn_150_int8.tflite"))
    it.allocate_tensors()
    return it


def int8_predict(it, batch_f32):
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    s, z = inp["quantization"]
    os_, oz = out["quantization"]
    preds = []
    for x in batch_f32:
        q = np.round(x[None] / s + z).clip(-128, 127).astype(inp["dtype"])
        it.set_tensor(inp["index"], q)
        it.invoke()
        p = (it.get_tensor(out["index"])[0].astype("float32") - oz) * os_
        preds.append(int(np.argmax(p)))
    return np.array(preds)


def load(path):
    return np.asarray(Image.open(path).convert("RGB").resize((150, 150)), dtype="float32") / 255.0


def shift(img, dx, dy):
    a = np.asarray(img)
    return Image.fromarray(np.roll(np.roll(a, dy, axis=0), dx, axis=1))


PERTURB = {
    "original": lambda im: im,
    "shift_x_10px": lambda im: shift(im, 10, 0),
    "shift_y_10px": lambda im: shift(im, 0, 10),
    "rotate_5deg": lambda im: im.rotate(5, resample=Image.BILINEAR),
    "zoom_in_10pct": lambda im: im.crop((8, 8, 142, 142)).resize((150, 150)),
    "brightness_x0.6": lambda im: ImageEnhance.Brightness(im).enhance(0.6),
    "brightness_x1.4": lambda im: ImageEnhance.Brightness(im).enhance(1.4),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--robust-n", type=int, default=4000)
    args = ap.parse_args()
    import tensorflow as tf

    base = Path(args.dataset)
    items = []
    for line in open(base / "LABELS" / "test.txt"):
        line = line.strip()
        if line:
            p, y = line.rsplit(" ", 1)
            items.append((p, int(y)))
    OUT.mkdir(parents=True, exist_ok=True)

    keras_model = tf.keras.models.load_model(ROOT / "model" / "parking_cnn_150.keras")
    it = load_int8()

    rows = []
    t0 = time.time()
    for i in range(0, len(items), 256):
        chunk = items[i:i + 256]
        x = np.stack([load(base / "PATCHES" / p) for p, _ in chunk])
        pf = keras_model.predict(x, verbose=0).argmax(1)
        pi = int8_predict(it, x)
        for (p, y), a, b in zip(chunk, pf, pi):
            rows.append((p, y, int(a), int(b)))
        if (i // 256) % 20 == 0:
            print(f"{i}/{len(items)}  {time.time() - t0:.0f}s", flush=True)

    y = np.array([r[1] for r in rows]); pf = np.array([r[2] for r in rows]); pi = np.array([r[3] for r in rows])
    summary = {
        "images": len(rows),
        "float_accuracy": round(float((pf == y).mean()), 4),
        "int8_accuracy": round(float((pi == y).mean()), 4),
        "float_int8_agreement": round(float((pf == pi).mean()), 4),
    }

    def group(key_fn):
        g = defaultdict(lambda: [0, 0])
        for p, yy, _, b in rows:
            k = key_fn(p)
            g[k][0] += 1
            g[k][1] += int(b == yy)
        return {k: {"images": n, "accuracy": round(c / n, 4)} for k, (n, c) in sorted(g.items())}

    summary["per_weather"] = group(lambda p: p.split("/")[0])
    summary["per_camera"] = group(lambda p: re.search(r"camera(\d+)", p).group(0))

    # camera-shift / lighting robustness on a fixed random subset
    random.seed(SEED)
    sub = random.sample(items, min(args.robust_n, len(items)))
    robust = {}
    for name, fn in PERTURB.items():
        xs, ys = [], []
        for p, yy in sub:
            im = Image.open(base / "PATCHES" / p).convert("RGB").resize((150, 150))
            xs.append(np.asarray(fn(im), dtype="float32") / 255.0)
            ys.append(yy)
        pred = int8_predict(it, np.stack(xs))
        robust[name] = round(float((pred == np.array(ys)).mean()), 4)
        print(name, robust[name], flush=True)
    summary["robustness_int8"] = {"subset_images": len(sub), "accuracy": robust}

    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
