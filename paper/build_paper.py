"""Fills template.html with the measured numbers and prints paper.pdf with headless Chrome.

    python paper/build_paper.py [--chrome PATH_TO_CHROME]
"""
import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "demo" / "results"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def pct(x, d=2):
    return f"{x * 100:.{d}f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrome", default=CHROME)
    args = ap.parse_args()

    full = json.loads((RES / "metrics_full_test.json").read_text())
    ana = json.loads((RES / "analysis" / "analysis_summary.json").read_text())
    cm = full["confusion_matrix"]
    fp, fn = cm["true_empty"]["pred_occupied"], cm["true_occupied"]["pred_empty"]
    n_empty = sum(cm["true_empty"].values())
    n_occ = sum(cm["true_occupied"].values())

    weather = ana["per_weather"]
    w_lo = min(weather.items(), key=lambda kv: kv[1]["accuracy"])
    w_hi = max(weather.items(), key=lambda kv: kv[1]["accuracy"])
    cams = ana["per_camera"]
    c_lo = min(cams.items(), key=lambda kv: kv[1]["accuracy"])
    c_hi = max(cams.items(), key=lambda kv: kv[1]["accuracy"])

    rob = ana["robustness_int8"]["accuracy"]
    base = rob["original"]
    rows = []
    for name, acc in rob.items():
        delta = "&ndash;" if name == "original" else f"{(acc - base) * 100:+.2f} pt"
        rows.append(f"<tr><td>{name.replace('_', ' ')}</td><td class='n'>{pct(acc)}</td><td class='n'>{delta}</td></tr>")
    worst = min(((k, v) for k, v in rob.items() if k != "original"), key=lambda kv: kv[1])
    drops = {k: base - v for k, v in rob.items() if k != "original"}
    small = [k for k, d in drops.items() if d < 0.01]
    rot = rob["rotate_5deg"]
    others = {k: v for k, v in rob.items() if k not in ("original", "rotate_5deg")}
    robust_sentence = (
        f"Shifts, zoom and brightness changes cost between {(base - max(others.values())) * 100:.1f} and "
        f"{(base - min(others.values())) * 100:.1f} points. The 5&deg; rotation was the exception: accuracy fell from "
        f"{pct(base)} to {pct(rot)}. Part of that is probably an artefact of our test, because rotating a patch leaves "
        f"black corners that the model never saw during training (Keras augmentation fills them by reflection), "
        f"but we cannot rule out a genuine weakness to tilt. We treat it as an open problem."
    )

    vals = {
        "n": f"{ana['images']:,}",
        "int8_acc_pct": pct(ana["int8_accuracy"]),
        "float_acc_pct": pct(ana["float_accuracy"]),
        "agree_pct": pct(ana["float_int8_agreement"]),
        "prec_pct": pct(full["precision_occupied"]),
        "rec_pct": pct(full["recall_occupied"]),
        "f1_pct": pct(full["f1_occupied"]),
        "fp": f"{fp:,}", "fn": f"{fn:,}",
        "fpr_pct": pct(fp / n_empty), "fnr_pct": pct(fn / n_occ),
        "robust_n": f"{ana['robustness_int8']['subset_images']:,}",
        "weather_sentence": (f"Accuracy is highest on {w_hi[0].lower()} days ({pct(w_hi[1]['accuracy'])}) and lowest "
                             f"on {w_lo[0].lower()} days ({pct(w_lo[1]['accuracy'])})."),
        "camera_sentence": (f"Across viewpoints it ranges from {pct(c_lo[1]['accuracy'])} ({c_lo[0]}) to "
                            f"{pct(c_hi[1]['accuracy'])} ({c_hi[0]}), so a few angles are clearly harder than others."),
        "robust_sentence": robust_sentence,
        "ROBUST_ROWS": "\n".join(rows),
    }
    html = (HERE / "template.html").read_text(encoding="utf-8")
    for k, v in vals.items():
        html = html.replace("{{" + k + "}}", v)
    assert "{{" not in html, "unfilled placeholder"
    out_html = HERE / "paper.html"
    out_html.write_text(html, encoding="utf-8")

    pdf = HERE / "paper.pdf"
    subprocess.run([args.chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", out_html.as_uri()], check=True, timeout=120)
    print("wrote", pdf)


if __name__ == "__main__":
    main()
