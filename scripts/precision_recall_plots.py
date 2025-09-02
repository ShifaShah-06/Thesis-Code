import matplotlib.pyplot as plt
import numpy as np

types = ["Inductive", "Deductive", "Abductive", "Analogical"]

# F1 data
f1_global = [0.880, 0.772, 0.877, 0.714]
f1_local  = [0.840, 0.767, 0.808, 0.698]
macro_global = sum(f1_global)/len(f1_global)
macro_local  = sum(f1_local)/len(f1_local)

# --- styling ---
BAR_WIDTH = 0.28  # thinner bars
COLORS = {"Global": "#1f5b86", "Local": "#65ad90"} #global = green, local = pink
EDGE = {"edgecolor": "black", "linewidth": 0.6}

x = np.arange(len(types))

fig, ax = plt.subplots(figsize=(8, 4.5))
bars1 = ax.bar(x - BAR_WIDTH/2, f1_global, BAR_WIDTH,
               label="Global", color=COLORS["Global"], **EDGE)
bars2 = ax.bar(x + BAR_WIDTH/2, f1_local,  BAR_WIDTH,
               label="Local",  color=COLORS["Local"], **EDGE)

ax.set_title("Classifier Performance (F1) by Reasoning Type - Global vs Local", pad=8)
ax.set_ylabel("F1 score")
ax.set_xticks(x)
ax.set_xticklabels(types)
ax.set_ylim(0, 1.0)
ax.grid(axis="y", linestyle=":", alpha=0.6)
ax.legend(frameon=False)

# macro-average lines (match series colors, lighter)
ax.axhline(macro_global, linestyle="--", linewidth=1, alpha=0.8, color=COLORS["Global"])
ax.text(len(types)-0.4, macro_global+0.01, f"Global mean ≈ {macro_global:.3f}",
        ha="right", va="bottom")
ax.axhline(macro_local, linestyle="--", linewidth=1, alpha=0.8, color=COLORS["Local"])
ax.text(len(types)-0.4, macro_local+0.01, f"Local mean ≈ {macro_local:.3f}",
        ha="right", va="bottom")

# annotate bars
for bars in (bars1, bars2):
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.2f}", xy=(b.get_x() + b.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("fig_classifier_F1_by_type_global_vs_local.png", dpi=300)
plt.show()


############################################################################################


# Precision/Recall values (from your tables)
import numpy as np
import matplotlib.pyplot as plt

types = ["Inductive","Deductive","Abductive","Analogical"]

# --- correct the inputs ---
prec_global = [0.882, 0.905, 0.823, 0.882]
rec_global  = [0.877, 0.673, 0.939, 0.600]
prec_local  = [0.983, 0.907, 0.748, 0.771]
rec_local   = [0.732, 0.665, 0.878, 0.638]

# simple label dodger to enforce a minimum vertical gap
def dodge_y(y_vals, min_gap=0.03, low=0.02, high=0.98):
    y = np.array(y_vals, float)
    order = np.argsort(y)              # bottom -> top
    y_adj = y.copy()
    last = -1.0
    for i in order:
        v = max(y_adj[i], last + min_gap) if last > -1 else y_adj[i]
        v = min(max(v, low), high)
        y_adj[i] = v
        last = v
    # compress if pushed out of bounds
    if y_adj.max() > high:
        y_adj -= (y_adj.max() - high)
    if y_adj.min() < low:
        y_adj += (low - y_adj.min())
    return y_adj

left_labels_y  = dodge_y(prec_global, min_gap=0.035)
right_labels_y = dodge_y(rec_local,  min_gap=0.035)

COL_G = "#1f5b86"   # global (blue)
COL_L = "#65ad90"   # local (orange)

fig, ax = plt.subplots(figsize=(8, 4.5))

# lines: Global (solid), Local (dashed)
for i, t in enumerate(types):
    ax.plot([0, 1], [prec_global[i], rec_global[i]],
            marker="o", linestyle="-",  linewidth=1.6, color=COL_G, alpha=0.9)
    ax.plot([0, 1], [prec_local[i],  rec_local[i]],
            marker="s", linestyle="--", linewidth=1.6, color=COL_L, alpha=0.9)

# left-side labels for Global (with leader lines)
for i, t in enumerate(types):
    ax.annotate(f"{t} (G)",
        xy=(0, prec_global[i]), xytext=(-0.08, left_labels_y[i]),
        ha="right", va="center", fontsize=9,
        arrowprops=dict(arrowstyle="-", lw=0.6, color="0.4"))

# right-side labels for Local (with leader lines)
for i, t in enumerate(types):
    ax.annotate(f"{t} (L)",
        xy=(1, rec_local[i]), xytext=(1.08, right_labels_y[i]),
        ha="left", va="center", fontsize=9,
        arrowprops=dict(arrowstyle="-", lw=0.6, color="0.4"))

ax.set_xlim(-0.2, 1.2)
ax.set_ylim(0, 1.0)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Precision", "Recall"])
ax.set_ylabel("Score")
ax.set_title("Precision vs Recall by Reasoning Type (Global = solid, Local = dashed)", pad=8)
ax.grid(axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("fig_classifier_precision_recall_by_type.png", dpi=300)
plt.show()
