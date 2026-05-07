#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""Clustered bar chart: Faithfulness and Completeness across T1 and Exec.
Four bars per cluster: dark/solid = BF16, light/hatched = AWQ (B&W-safe).
Error bars = ±1 SE."""

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

ROOT   = Path(__file__).parent / "results"
TABLES = Path(__file__).parent / "tables"
TABLES.mkdir(exist_ok=True)

MODELS = {
    "9B-BF16": {"judgements": ROOT/"judgements/local-Qwen3.6-27B-FP8_Qwen__Qwen3.5-9B.json",
                "summaries":  ROOT/"summaries/Qwen__Qwen3.5-9B.json",
                "metrics":    ROOT/"metrics/Qwen__Qwen3.5-9B.json"},
    "9B-AWQ":  {"judgements": ROOT/"judgements/local-Qwen3.6-27B-FP8_QuantTrio__Qwen3.5-9B-AWQ.json",
                "summaries":  ROOT/"summaries/QuantTrio__Qwen3.5-9B-AWQ.json",
                "metrics":    ROOT/"metrics/QuantTrio__Qwen3.5-9B-AWQ.json"},
    "4B-BF16": {"judgements": ROOT/"judgements/local-Qwen3.6-27B-FP8_Qwen__Qwen3.5-4B.json",
                "summaries":  ROOT/"summaries/Qwen__Qwen3.5-4B.json",
                "metrics":    ROOT/"metrics/Qwen__Qwen3.5-4B.json"},
    "4B-AWQ":  {"judgements": ROOT/"judgements/local-Qwen3.6-27B-FP8_QuantTrio__Qwen3.5-4B-AWQ.json",
                "summaries":  ROOT/"summaries/QuantTrio__Qwen3.5-4B-AWQ.json",
                "metrics":    ROOT/"metrics/QuantTrio__Qwen3.5-4B-AWQ.json"},
}
KEYS    = ["9B-BF16", "9B-AWQ", "4B-BF16", "4B-AWQ"]
T1_EXCL  = set()
ACT_EXCL = set()

# Dark = BF16, light = AWQ — same hue per model family
COLORS = {
    "9B-BF16": "#1a6fad",
    "9B-AWQ":  "#8ab8d8",
    "4B-BF16": "#c0392b",
    "4B-AWQ":  "#e08c87",
}
LABELS = {
    "9B-BF16": "9B BF16",
    "9B-AWQ":  "9B AWQ",
    "4B-BF16": "4B BF16",
    "4B-AWQ":  "4B AWQ",
}
# AWQ bars: family-colour edge + diagonal hatch → distinguishable in B&W print
EDGE  = {"9B-BF16": "white",   "9B-AWQ": "#1a6fad",
         "4B-BF16": "white",   "4B-AWQ": "#c0392b"}
HATCH = {"9B-BF16": None,      "9B-AWQ": "//",
         "4B-BF16": None,      "4B-AWQ": "//"}


def _score(key, kind, tier, metric, exclude):
    recs  = json.loads(MODELS[key]["summaries"].read_text())
    valid = {r["record_id"] for r in recs if r.get("tier") == tier and not r.get("error")}
    jrecs = json.loads(MODELS[key]["judgements"].read_text())
    by_mtg = defaultdict(list)
    for r in jrecs:
        if r["kind"] == kind and r.get("payload") and not r.get("error"):
            if r["record_id"] not in valid:
                continue
            v = r["payload"].get(metric)
            if v is not None:
                by_mtg[r["transcript_id"]].append(v)
    vals = [statistics.mean(vs) for tid, vs in by_mtg.items() if tid not in exclude]
    n = len(vals)
    return statistics.mean(vals), statistics.stdev(vals) / n ** 0.5


def _wall_clock(key, tier, excl=frozenset()):
    recs = json.loads(MODELS[key]["summaries"].read_text())
    by_mtg = defaultdict(float)
    for r in recs:
        if (r.get("tier") == tier and not r.get("is_warmup") and not r.get("error")
                and r.get("transcript_id") not in excl):
            by_mtg[r["transcript_id"]] += r["wall_time_s"]
    vals = list(by_mtg.values())
    return statistics.mean(vals), statistics.stdev(vals) / len(vals) ** 0.5

wall = {key: {"t1":   _wall_clock(key, tier=1, excl=T1_EXCL),
              "exec": _wall_clock(key, tier=0)}
        for key in KEYS}

# ── action F1 data ─────────────────────────────────────────────────────────
_idx        = json.loads((ROOT / "transcripts" / "index.json").read_text())
_bucket_map = {m["transcript_id"]: m["bucket"] for m in _idx["transcripts"]}
BUCKETS     = ["short", "medium", "long"]
BLABELS     = ["Short", "Medium", "Long"]

_f1_raw = {key: {k: v["f1"]
                 for k, v in json.loads(MODELS[key]["metrics"].read_text())
                                        ["action_judge"]["per_meeting"].items()}
           for key in KEYS}

def _f1_bucket_stats(key):
    out = {}
    for b in BUCKETS:
        vals = [_f1_raw[key][tid]
                for tid, bk in _bucket_map.items()
                if bk == b and tid not in ACT_EXCL and tid in _f1_raw[key]]
        n = len(vals)
        out[b] = (statistics.mean(vals), statistics.stdev(vals) / n**0.5)
    return out

f1_stats = {key: _f1_bucket_stats(key) for key in KEYS}

data = {key: {
    "t1_faith":   _score(key, "summary", 1, "faithfulness", T1_EXCL),
    "exec_faith": _score(key, "exec",    0, "faithfulness", set()),
    "t1_comp":    _score(key, "summary", 1, "completeness", T1_EXCL),
    "exec_comp":  _score(key, "exec",    0, "completeness", set()),
} for key in KEYS}

# ── layout constants ──────────────────────────────────────────────────────────
N       = len(KEYS)
BAR_W   = 0.18
SPACING = 0.03
CLUSTER = N * BAR_W + (N - 1) * SPACING   # total cluster width
GAP     = 0.5                              # space between T1 and Exec clusters
X_T1    = 0.0
X_EXEC  = CLUSTER + GAP
X_SHORT = 0.0
X_MED   = CLUSTER + GAP
X_LONG  = 2 * (CLUSTER + GAP)

# x position for bar i within a cluster at centre x0
def bar_x(x0, i):
    return x0 + i * (BAR_W + SPACING)

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(3.5, 4.8))
gs  = gridspec.GridSpec(5, 1, figure=fig,
                        height_ratios=[1.0, 1.0, 0.8, 0.28, 1.0], hspace=0.18, top=0.90)
ax0  = fig.add_subplot(gs[0])
ax1  = fig.add_subplot(gs[1], sharex=ax0, sharey=ax0)
ax_w = fig.add_subplot(gs[2], sharex=ax0)
# gs[3] is a blank spacer — hide all its elements
_ax_spacer = fig.add_subplot(gs[3])
_ax_spacer.set_visible(False)
ax_f = fig.add_subplot(gs[4])

PANELS = [
    (ax0, "t1_faith", "exec_faith", "Faithfulness"),
    (ax1, "t1_comp",  "exec_comp",  "Completeness"),
]

for ax, t1_key, exec_key, panel_label in PANELS:
    for i, key in enumerate(KEYS):
        for x0, metric_key in [(X_T1, t1_key), (X_EXEC, exec_key)]:
            mean, se = data[key][metric_key]
            x = bar_x(x0, i)
            ax.bar(x, mean, width=BAR_W,
                   color=COLORS[key], edgecolor=EDGE[key],
                   hatch=HATCH[key], linewidth=0.5, zorder=2)
            ax.errorbar(x, mean, yerr=se,
                        fmt="none", color="#444",
                        capsize=2.5, capthick=0.9, elinewidth=0.9, zorder=3)

    # 4.0 reference line
    ax.axhline(4.0, color="#bbbbbb", lw=0.8, ls="--", zorder=1)

    # panel label top-left
    ax.text(0.98, 0.97, panel_label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="right")

    ax.set_ylim(2, 5.4)
    ax.set_yticks([2, 3, 4, 5])
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False, labelbottom=False, labelsize=8)

ax0.set_ylabel("Score (0–5,  ±1 SE)", fontsize=9)

# x-axis group labels centred under each cluster
cluster_centre_t1   = X_T1   + (CLUSTER - BAR_W) / 2 + BAR_W / 2
cluster_centre_exec = X_EXEC + (CLUSTER - BAR_W) / 2 + BAR_W / 2
ax_w.set_xticks([cluster_centre_t1, cluster_centre_exec])
ax_w.set_xticklabels(["Primary summaries", "Executive summary"], fontsize=9)
ax_w.tick_params(bottom=False)

# ── wall clock line panel — single series, one point per bar above ────────────
xs  = [bar_x(X_T1,   i) for i in range(N)] + [bar_x(X_EXEC, i) for i in range(N)]
ys  = [wall[k]["t1"][0]   for k in KEYS]   + [wall[k]["exec"][0] for k in KEYS]
ses = [wall[k]["t1"][1]   for k in KEYS]   + [wall[k]["exec"][1] for k in KEYS]

ax_w.plot(xs, ys, color="#555", lw=1.4, zorder=2)
ax_w.errorbar(xs, ys, yerr=ses, fmt="o", color="#555",
              markersize=4, capsize=2.5, capthick=0.9, elinewidth=0.9, zorder=3)

y_lo = max(0, min(v - e for v, e in zip(ys, ses)) - 5)
y_hi = max(v + e for v, e in zip(ys, ses)) + 10
ax_w.set_ylim(y_lo, y_hi)
ax_w.yaxis.set_major_locator(plt.MaxNLocator(3, integer=True))
ax_w.set_ylabel("s  (±SE)", fontsize=8)
ax_w.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
ax_w.set_axisbelow(True)
ax_w.spines["top"].set_visible(False)
ax_w.spines["right"].set_visible(False)
ax_w.spines["bottom"].set_visible(False)
ax_w.tick_params(labelsize=8)
ax_w.text(0.98, 0.97, "Wall clock", transform=ax_w.transAxes,
          fontsize=9, fontweight="bold", va="top", ha="right")

# ── action F1 bar panel ───────────────────────────────────────────────────────
for i, key in enumerate(KEYS):
    for x0, b in zip([X_SHORT, X_MED, X_LONG], BUCKETS):
        mean, se = f1_stats[key][b]
        x = bar_x(x0, i)
        ax_f.bar(x, mean, width=BAR_W,
                 color=COLORS[key], edgecolor=EDGE[key],
                 hatch=HATCH[key], linewidth=0.5, zorder=2)
        ax_f.errorbar(x, mean, yerr=se,
                      fmt="none", color="#444",
                      capsize=2.5, capthick=0.9, elinewidth=0.9, zorder=3)

_f1_pairs = [(f1_stats[k][b][0], f1_stats[k][b][1]) for k in KEYS for b in BUCKETS]
_f1_y_lo  = max(0.0, min(m - e for m, e in _f1_pairs) - 0.04)
ax_f.set_ylim(_f1_y_lo, 1.07)
ax_f.yaxis.set_major_locator(plt.MaxNLocator(4, prune="lower"))
ax_f.set_ylabel("Action F1  (±SE)", fontsize=8)

_cc_f1 = [x0 + (bar_x(0, N - 1) / 2) for x0 in [X_SHORT, X_MED, X_LONG]]
ax_f.set_xticks(_cc_f1)
ax_f.set_xticklabels(BLABELS, fontsize=9)
ax_f.tick_params(bottom=False)
ax_f.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
ax_f.set_axisbelow(True)
ax_f.spines["top"].set_visible(True)
ax_f.spines["top"].set_linewidth(0.8)
ax_f.spines["top"].set_color("#aaaaaa")
ax_f.spines["right"].set_visible(False)
ax_f.spines["bottom"].set_visible(False)
ax_f.text(0.98, 0.97, "Action F1", transform=ax_f.transAxes,
          fontsize=9, fontweight="bold", va="top", ha="right")

# legend above both panels
handles = [mpatches.Patch(facecolor=COLORS[k], edgecolor=EDGE[k],
                           hatch=HATCH[k], label=LABELS[k])
           for k in KEYS]
fig.legend(handles=handles, loc="upper center", ncol=4,
           fontsize=7.5, frameon=False,
           bbox_to_anchor=(0.54, 0.99),
           columnspacing=0.8, handletextpad=0.4)

fig.align_ylabels([ax0, ax_w, ax_f])
fig.savefig(TABLES / "plot_quality_bars.png", dpi=150, bbox_inches="tight")
print("Saved:", TABLES / "plot_quality_bars.png")
