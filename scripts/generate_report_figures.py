"""Generate matplotlib figures for the final report.

Outputs land in figures/report/. Run from project root:
    python scripts/generate_report_figures.py
"""

import csv
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PILOT_SUMMARY = "outputs/pilot/pilot_summary.json"
MAIN_SUMMARY = "outputs/main_oscar/llava15_main_summary.json"
MAIN_DIR = "outputs/main_oscar/llava15"
OUT_DIR = "figures/report"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# Figure 1: Pilot model comparison
# ----------------------------------------------------------------------------

def fig_pilot(out_path):
    rows = load_json(PILOT_SUMMARY)
    models = [r["model"] for r in rows]
    accs = [r["accuracy"] for r in rows]

    pretty = {"blip2": "BLIP-2 (2.7B)", "qwen": "Qwen2.5-VL (3B)", "llava15": "LLaVA-1.5 (7B)"}
    labels = [pretty.get(m, m) for m in models]
    colors = ["#9bbcd8", "#e6a55a", "#3b7a57"]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(labels, accs, color=colors, edgecolor="black", linewidth=0.6)
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Exact-match Accuracy")
    ax.set_title("Pilot: zero-shot accuracy on 100 TextVQA validation samples")
    ax.set_ylim(0, max(accs) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"saved {out_path}")


# ----------------------------------------------------------------------------
# Figure 2: Prompt comparison (multiple metrics, grouped bars)
# ----------------------------------------------------------------------------

def fig_prompts(out_path):
    rows = load_json(MAIN_SUMMARY)
    # Order prompts so baseline / cot are first, optimized last
    order = [
        "baseline", "cot", "ocr_short", "minimal_answer",
        "constrained", "ocr_exact", "key_focus",
        "textvqa_final", "verbatim_ocr", "few_shot",
    ]
    rows_sorted = sorted(rows, key=lambda r: order.index(r["prompt"]))

    prompts = [r["prompt"] for r in rows_sorted]
    accs = [r["accuracy"] for r in rows_sorted]
    softs = [r["vqa_soft_accuracy"] for r in rows_sorted]
    subs = [r["substring_match"] for r in rows_sorted]
    judges = [r["llm_judge"] for r in rows_sorted]

    x = np.arange(len(prompts))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 1.5 * width, accs, width, label="Accuracy (exact)", color="#3b7a57")
    ax.bar(x - 0.5 * width, softs, width, label="VQA-soft (TextVQA official)", color="#7fb069")
    ax.bar(x + 0.5 * width, subs, width, label="Substring match", color="#e6a55a")
    ax.bar(x + 1.5 * width, judges, width, label="LLM-Judge (semantic)", color="#9bbcd8")

    ax.set_xticks(x)
    ax.set_xticklabels(prompts, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Prompt comparison on 200 validation samples (LLaVA-1.5-7B)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"saved {out_path}")


# ----------------------------------------------------------------------------
# Figure 3: Per-category accuracy bar chart for the best prompt
# ----------------------------------------------------------------------------

def fig_categories(out_path, prompt_name="constrained"):
    cat_path = os.path.join(MAIN_DIR, prompt_name, "category_breakdown.json")
    rows = load_json(cat_path)

    rows_sorted = sorted(rows, key=lambda r: -r["accuracy"])

    cats = [r["category"] for r in rows_sorted]
    counts = [r["count"] for r in rows_sorted]
    accs = [r["accuracy"] for r in rows_sorted]
    softs = [r["vqa_soft"] for r in rows_sorted]
    subs = [r["substring"] for r in rows_sorted]
    judges = [r["llm_judge"] for r in rows_sorted]

    x = np.arange(len(cats))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5 * width, accs, width, label="Accuracy", color="#3b7a57")
    ax.bar(x - 0.5 * width, softs, width, label="VQA-soft", color="#7fb069")
    ax.bar(x + 0.5 * width, subs, width, label="Substring", color="#e6a55a")
    ax.bar(x + 1.5 * width, judges, width, label="LLM-Judge", color="#9bbcd8")

    labels = [f"{c}\n(n={n})" for c, n in zip(cats, counts)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(f"Per-category breakdown ({prompt_name} prompt, 200 samples)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"saved {out_path}")


# ----------------------------------------------------------------------------
# Figure 4: Heatmap of accuracy across (prompt × category)
# ----------------------------------------------------------------------------

def fig_heatmap(out_path):
    order = [
        "baseline", "cot", "ocr_short", "minimal_answer",
        "constrained", "ocr_exact", "key_focus",
        "textvqa_final", "verbatim_ocr", "few_shot",
    ]
    cat_order = ["yes_no", "color", "brand", "date_year", "other",
                 "number", "text_read", "price", "time"]

    matrix = []
    for prompt in order:
        cat_path = os.path.join(MAIN_DIR, prompt, "category_breakdown.json")
        cats = {r["category"]: r["accuracy"] for r in load_json(cat_path)}
        matrix.append([cats.get(c, np.nan) for c in cat_order])
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cat_order)))
    ax.set_xticklabels(cat_order, rotation=30, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("Question category")
    ax.set_ylabel("Prompt variant")
    ax.set_title("Per-category accuracy across all 10 prompts (LLaVA-1.5)")

    for i in range(len(order)):
        for j in range(len(cat_order)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}",
                        ha="center", va="center",
                        color="black" if 0.3 < v < 0.85 else "white",
                        fontsize=8)

    fig.colorbar(im, ax=ax, label="Accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"saved {out_path}")


# ----------------------------------------------------------------------------
# Figure 5: Pipeline diagram (simple block diagram done in matplotlib)
# ----------------------------------------------------------------------------

def fig_pipeline(out_path):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    blocks = [
        (0.5, "TextVQA\nval split\n(5k QA pairs)", "#9bbcd8"),
        (3.0, "Prompt\nbuilder\n(10 templates)", "#e6a55a"),
        (5.5, "LLaVA-1.5-7B\n(zero-shot,\nfp16, GPU)", "#3b7a57"),
        (8.0, "Predictions\n+ latency log", "#cccccc"),
        (10.5, "Metrics:\nAcc / VQA-soft /\nBLEU / METEOR /\nROUGE / LLM-Judge", "#7fb069"),
    ]
    for x, label, color in blocks:
        rect = plt.Rectangle((x, 1), 1.5, 2, facecolor=color, edgecolor="black", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + 0.75, 2, label, ha="center", va="center", fontsize=9)

    # Arrows
    for i in range(len(blocks) - 1):
        x_start = blocks[i][0] + 1.55
        x_end = blocks[i + 1][0] - 0.05
        ax.annotate("", xy=(x_end, 2), xytext=(x_start, 2),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="black"))

    # Branching to per-category analysis
    ax.annotate("", xy=(11.25, 0.6), xytext=(11.25, 0.95),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
    ax.text(11.25, 0.35, "+ per-category\nbreakdown", ha="center", va="center", fontsize=8.5)

    ax.set_title("Evaluation pipeline", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    ensure_dir(OUT_DIR)
    fig_pilot(os.path.join(OUT_DIR, "fig1_pilot.png"))
    fig_prompts(os.path.join(OUT_DIR, "fig2_prompts.png"))
    fig_categories(os.path.join(OUT_DIR, "fig3_categories.png"))
    fig_heatmap(os.path.join(OUT_DIR, "fig4_heatmap.png"))
    fig_pipeline(os.path.join(OUT_DIR, "fig5_pipeline.png"))


if __name__ == "__main__":
    main()
