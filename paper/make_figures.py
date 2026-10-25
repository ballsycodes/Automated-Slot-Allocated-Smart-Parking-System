"""Builds the figures for the manuscript from the saved result files.

    python paper/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "demo" / "results"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
MAROON, GREY, LIGHT = "#a4123f", "#555555", "#f3eef0"
plt.rcParams.update({"font.family": "serif", "font.size": 9})


def box(ax, x, y, w, h, text, fc=LIGHT, ec=GREY, bold=False, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, dashed=False):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.0, ls="--" if dashed else "-"))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    ax.set_xlim(0, 14.6); ax.set_ylim(0, 3.2); ax.axis("off")
    steps = ["OV3660\nRGB565\n320x240", "Resize\n150x150", "RGB565 to\nRGB888", "INT8\nquantize",
             "TFLite Micro\nCNN", "Dequantize\n+ argmax", "EMPTY /\nOCCUPIED"]
    w, gap = 1.75, 0.34
    for i, s in enumerate(steps):
        x = 0.1 + i * (w + gap)
        box(ax, x, 1.0, w, 1.4, s, fc="#ffffff" if i not in (4, 6) else LIGHT, bold=i in (4, 6))
        if i < len(steps) - 1:
            arrow(ax, x + w, 1.7, x + w + gap, 1.7)
    ax.text(7.2, 0.35, "All stages run on the ESP32-S3; the tensor arena and frame buffers live in PSRAM",
            ha="center", fontsize=8, style="italic", color=GREY)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "fig_pipeline.png", dpi=220)
    plt.close(fig)


def fig_cnn():
    fig, ax = plt.subplots(figsize=(7.2, 1.7))
    ax.set_xlim(0, 14.4); ax.set_ylim(0, 3.2); ax.axis("off")
    layers = ["Input\n150x150x3", "Conv 3x3\n16, ReLU", "MaxPool\n2x2", "Conv 3x3\n32, ReLU", "MaxPool\n2x2",
              "Conv 3x3\n64, ReLU", "Global\nAvgPool", "Dense 32\nReLU", "Dropout\n0.3", "Dense 2\nsoftmax"]
    w, gap = 1.22, 0.24
    for i, s in enumerate(layers):
        x = 0.1 + i * (w + gap)
        conv = "Conv" in s
        box(ax, x, 0.9, w, 1.5, s, fc=LIGHT if conv else "#ffffff", ec=MAROON if conv else GREY, fs=7.5)
        if i < len(layers) - 1:
            arrow(ax, x + w, 1.65, x + w + gap, 1.65)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "fig_cnn.png", dpi=220)
    plt.close(fig)


def fig_system():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 14.4); ax.set_ylim(0, 6); ax.axis("off")
    box(ax, 0.2, 3.9, 3.4, 1.5, "ESP32-S3 + OV3660\nslot node A\n(TinyML occupancy)", fc="#ffffff")
    box(ax, 0.2, 1.1, 3.4, 1.5, "ESP32-S3 + OV3660\nslot node B\n(TinyML occupancy)", fc="#ffffff")
    box(ax, 5.2, 2.3, 4.0, 2.2, "Raspberry Pi 3A+\nFlask API, SQLite,\nallocation engine, QR", fc=LIGHT, ec=MAROON, bold=True)
    box(ax, 10.9, 4.0, 3.3, 1.4, "Driver\nQR scan, web page, SMS", fc="#ffffff")
    box(ax, 10.9, 1.1, 3.3, 1.4, "Supabase\ncloud sync", fc="#ffffff")
    arrow(ax, 3.6, 4.6, 5.2, 3.9, dashed=True); arrow(ax, 3.6, 1.9, 5.2, 2.9, dashed=True)
    ax.text(4.0, 3.9, "LoRa\n(planned)", fontsize=7.5, color=GREY, ha="center")
    arrow(ax, 9.2, 3.9, 10.9, 4.6); arrow(ax, 9.2, 2.9, 10.9, 1.9)
    ax.text(0.2, 0.3, "Solid boxes: designed system. Dashed links: not yet built. Implemented so far: the camera node.",
            fontsize=7.5, style="italic", color=GREY)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "fig_system.png", dpi=220)
    plt.close(fig)


def fig_confusion():
    m = json.loads((RES / "metrics_full_test.json").read_text())["confusion_matrix"]
    vals = [[m["true_empty"]["pred_empty"], m["true_empty"]["pred_occupied"]],
            [m["true_occupied"]["pred_empty"], m["true_occupied"]["pred_occupied"]]]
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    ax.imshow(vals, cmap="Purples")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{vals[i][j]:,}", ha="center", va="center",
                    color="white" if vals[i][j] > 9000 else "black", fontsize=10)
    ax.set_xticks([0, 1], ["Empty", "Occupied"]); ax.set_yticks([0, 1], ["Empty", "Occupied"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(OUT / "fig_confusion.png", dpi=220)
    plt.close(fig)


def fig_groups():
    a = json.loads((RES / "analysis" / "analysis_summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1, 2.2]})
    w = a["per_weather"]
    axes[0].bar(list(w.keys()), [v["accuracy"] * 100 for v in w.values()], color=MAROON)
    axes[0].set_ylim(90, 100); axes[0].set_ylabel("Accuracy (%)"); axes[0].set_title("By weather", fontsize=9)
    c = a["per_camera"]
    axes[1].bar([k.replace("camera", "cam ") for k in c], [v["accuracy"] * 100 for v in c.values()], color=MAROON)
    axes[1].set_ylim(90, 100); axes[1].set_title("By camera viewpoint", fontsize=9)
    for ax in axes:
        ax.grid(axis="y", alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_groups.png", dpi=220)
    plt.close(fig)


def fig_robust():
    a = json.loads((RES / "analysis" / "analysis_summary.json").read_text())["robustness_int8"]["accuracy"]
    names = list(a.keys())
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    bars = ax.bar([n.replace("_", "\n") for n in names], [v * 100 for v in a.values()],
                  color=[GREY if n == "original" else MAROON for n in names])
    ax.set_ylim(max(0, min(a.values()) * 100 - 8), 100); ax.set_ylabel("Accuracy (%)")
    for b, v in zip(bars, a.values()):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 0.3, f"{v * 100:.1f}", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_robust.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline(); fig_cnn(); fig_system(); fig_confusion()
    if (RES / "analysis" / "analysis_summary.json").exists():
        fig_groups(); fig_robust()
    print("figures written to", OUT)
