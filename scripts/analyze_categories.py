"""Compute per-category accuracy breakdown for a saved predictions.json."""

import argparse
import csv
import json
import os

from src.eval.categorize import per_category_breakdown
from src.eval.metrics import (
    exact_match,
    vqa_soft_accuracy,
    substring_match,
    best_char_similarity,
    llm_judge_score,
)


METRICS = {
    "accuracy": exact_match,
    "vqa_soft": vqa_soft_accuracy,
    "substring": substring_match,
    "char_sim": best_char_similarity,
    "llm_judge": llm_judge_score,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-category accuracy analysis on saved predictions."
    )
    p.add_argument("--input", required=True, help="Path to predictions.json")
    p.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write per-category breakdown files.",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["accuracy", "vqa_soft", "substring", "char_sim"],
        choices=list(METRICS),
        help="Which row-level metrics to break down by category.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        rows = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    summary = {}
    for metric_name in args.metrics:
        breakdown = per_category_breakdown(rows, METRICS[metric_name])
        summary[metric_name] = breakdown

    # Build a wide table: row=category, col=metric.
    categories = sorted({c for m in summary.values() for c in m})
    table = []
    for cat in categories:
        # All metrics share the same row count for a category, take from first.
        first = next(iter(summary.values()))
        count = first.get(cat, {}).get("count", 0)
        row = {"category": cat, "count": count}
        for metric_name in args.metrics:
            row[metric_name] = round(
                summary[metric_name].get(cat, {}).get("score", 0.0), 4
            )
        table.append(row)

    json_path = os.path.join(args.output_dir, "category_breakdown.json")
    csv_path = os.path.join(args.output_dir, "category_breakdown.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)

    if table:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            writer.writeheader()
            writer.writerows(table)

    print(f"Saved breakdown to: {json_path}")
    print(f"Saved breakdown to: {csv_path}")
    print()
    print(json.dumps(table, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
