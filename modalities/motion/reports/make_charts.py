"""
Generates all charts for the model-comparison analysis report.
Palette + mark specs follow the project's dataviz skill (validated default palette).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUT, exist_ok=True)

# ── Palette (light mode, validated) ─────────────────────────────────────────
BLUE    = "#2a78d6"
AQUA    = "#1baf7a"
YELLOW  = "#eda100"
GREEN   = "#008300"
VIOLET  = "#4a3aa7"
RED     = "#e34948"
MAGENTA = "#e87ba4"
ORANGE  = "#eb6834"

GOOD     = "#0ca30c"
CRITICAL = "#d03b3b"

SURFACE      = "#fcfcfb"
PRIMARY_INK  = "#0b0b0b"
SECOND_INK   = "#52514e"
MUTED_INK    = "#898781"
GRIDLINE     = "#e1e0d9"
BASELINE_AX  = "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": PRIMARY_INK,
    "axes.edgecolor": BASELINE_AX,
    "axes.labelcolor": SECOND_INK,
    "xtick.color": MUTED_INK,
    "ytick.color": MUTED_INK,
    "font.size": 11,
})


def rounded_bar(ax, x, height, width, color, radius=0.06):
    """Bar with a rounded top edge, square baseline — per mark spec."""
    if height <= 0:
        return
    box = FancyBboxPatch(
        (x - width / 2, 0), width, height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0, facecolor=color, mutation_aspect=1,
    )
    ax.add_patch(box)


def style_axes(ax, ymax=None, pct=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE_AX)
    ax.spines["bottom"].set_linewidth(1)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    if pct:
        ax.yaxis.set_major_formatter(lambda v, _: f"{int(v)}%")


# ─────────────────────────────────────────────────────────────────────────
# Chart 1 — NTU validation-accuracy progression (single series -> one hue)
# ─────────────────────────────────────────────────────────────────────────
labels = ["6-class\nrun 1", "6-class\nrun 2", "6-class\nfinal\n(tuned)", "4-class\nbaseline"]
values = [92.484, 93.950, 95.926, 96.705]

fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
xs = np.arange(len(labels))
width = 0.5
for x, v in zip(xs, values):
    rounded_bar(ax, x, v, width, BLUE, radius=0.10)
    ax.text(x, v + 1.8, f"{v:.1f}%", ha="center", va="bottom",
            fontsize=10.5, color=PRIMARY_INK, fontweight="bold")
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=9.5)
style_axes(ax, ymax=108)
ax.set_title("NTU Validation Accuracy by Training Run", fontsize=13, fontweight="bold",
             color=PRIMARY_INK, loc="left", pad=14)
ax.set_xlim(-0.6, len(labels) - 0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart_ntu_progression.png"))
plt.close()

# ─────────────────────────────────────────────────────────────────────────
# Chart 2 — Real-world before/after (status colors: critical vs good)
# ─────────────────────────────────────────────────────────────────────────
cats = ["Clip-level accuracy", "Window-level accuracy"]
baseline_vals = [27.4, 30.5]
finetuned_vals = [82.6, 74.6]

fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=200)
x = np.arange(len(cats))
w = 0.32
for i, (b, f) in enumerate(zip(baseline_vals, finetuned_vals)):
    rounded_bar(ax, x[i] - w / 2 - 0.02, b, w, CRITICAL, radius=0.09)
    rounded_bar(ax, x[i] + w / 2 + 0.02, f, w, GOOD, radius=0.09)
    ax.text(x[i] - w / 2 - 0.02, b + 2, f"{b:.1f}%", ha="center", fontsize=10.5,
            color=PRIMARY_INK, fontweight="bold")
    ax.text(x[i] + w / 2 + 0.02, f + 2, f"{f:.1f}%", ha="center", fontsize=10.5,
            color=PRIMARY_INK, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=10.5)
style_axes(ax, ymax=100)
ax.set_title("Real-World Performance: NTU-only Baseline vs. Fine-Tuned\n(held-out test split, hri-multimodal-intent-v1.0.0)",
             fontsize=12.5, fontweight="bold", color=PRIMARY_INK, loc="left", pad=14)
handles = [plt.Rectangle((0, 0), 1, 1, color=CRITICAL), plt.Rectangle((0, 0), 1, 1, color=GOOD)]
ax.legend(handles, ["Baseline (NTU-only)", "Fine-tuned"], loc="upper left",
          frameon=False, fontsize=10)
ax.set_xlim(-0.55, len(cats) - 0.45)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart_realworld_before_after.png"))
plt.close()

# ─────────────────────────────────────────────────────────────────────────
# Chart 3 — Per-class Precision/Recall/F1, fine-tuned model, held-out test
# ─────────────────────────────────────────────────────────────────────────
classes = ["sitting", "standing", "walking", "stepping_back"]
precision = [0.877, 0.960, 0.690, 0.833]
recall    = [0.926, 0.960, 0.875, 0.333]
f1        = [0.901, 0.960, 0.772, 0.476]

fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=200)
x = np.arange(len(classes))
w = 0.24
series = [("Precision", precision, BLUE), ("Recall", recall, AQUA), ("F1-score", f1, YELLOW)]
offsets = [-w - 0.02, 0, w + 0.02]
for (name, vals, color), off in zip(series, offsets):
    for xi, v in zip(x, vals):
        rounded_bar(ax, xi + off, v * 100, w, color, radius=0.08)
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=10.5)
style_axes(ax, ymax=115)
ax.set_title("Fine-Tuned Model — Per-Class Metrics (held-out test split, clip-level)",
             fontsize=12.5, fontweight="bold", color=PRIMARY_INK, loc="left", pad=14)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in series]
ax.legend(handles, [n for n, _, _ in series], loc="upper right", ncol=3,
          frameon=False, fontsize=10)
ax.set_xlim(-0.6, len(classes) - 0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart_perclass_finetuned_test.png"))
plt.close()

# ─────────────────────────────────────────────────────────────────────────
# Chart 4 — Confusion matrices, baseline vs fine-tuned (small multiples,
# sequential single-hue heatmap)
# ─────────────────────────────────────────────────────────────────────────
cm_baseline = np.array([
    [26, 25, 0, 3],
    [7, 13, 0, 30],
    [10, 25, 8, 13],
    [5, 20, 0, 5],
])
cm_finetuned = np.array([
    [50, 0, 4, 0],
    [2, 48, 0, 0],
    [4, 1, 49, 2],
    [1, 1, 18, 10],
])
names_short = ["sitting", "standing", "walking", "stepping\nback"]

fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=200)
for ax, cm, title in zip(axes, [cm_baseline, cm_finetuned],
                          ["Baseline (NTU-only)", "Fine-tuned"]):
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = cm / row_sums
    im = ax.imshow(norm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue_seq", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#104281"]), vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(names_short, fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels(names_short, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10, color=SECOND_INK)
    ax.set_ylabel("True", fontsize=10, color=SECOND_INK)
    ax.set_title(title, fontsize=12, fontweight="bold", color=PRIMARY_INK, pad=10)
    for i in range(4):
        for j in range(4):
            val = cm[i, j]
            frac = norm[i, j]
            txt_color = "white" if frac > 0.55 else PRIMARY_INK
            ax.text(j, i, str(val), ha="center", va="center", fontsize=11,
                    color=txt_color, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
fig.suptitle("Confusion Matrices — Held-Out Test Split (clip-level, row-normalized shading)",
             fontsize=13, fontweight="bold", color=PRIMARY_INK, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart_confusion_matrices.png"), bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────
# Chart 5 — Fine-tune training-data class scarcity (single series -> one hue)
# ─────────────────────────────────────────────────────────────────────────
ft_classes = ["sitting", "standing", "walking", "stepping_back"]
clip_counts = [370, 192, 180, 74]

fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
xs = np.arange(len(ft_classes))
colors = [BLUE, BLUE, BLUE, CRITICAL]
for x, v, c in zip(xs, clip_counts, colors):
    rounded_bar(ax, x, v, 0.5, c, radius=0.10)
    ax.text(x, v + 8, str(v), ha="center", fontsize=10.5, color=PRIMARY_INK, fontweight="bold")
ax.set_xticks(xs)
ax.set_xticklabels(ft_classes, fontsize=10.5)
style_axes(ax, ymax=420, pct=False)
ax.set_ylabel("Training clips", fontsize=10)
ax.set_title("Fine-Tuning Training Data — Clips per Class\n(stepping_back is the scarcity bottleneck)",
             fontsize=12.5, fontweight="bold", color=PRIMARY_INK, loc="left", pad=14)
ax.set_xlim(-0.6, len(ft_classes) - 0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart_class_scarcity.png"))
plt.close()

print("All charts saved to", OUT)
for f in sorted(os.listdir(OUT)):
    print(" -", f)
