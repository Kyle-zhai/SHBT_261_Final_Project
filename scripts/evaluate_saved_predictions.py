"""Recompute metrics on a saved predictions.json without re-running the model.

Outputs a JSON + CSV with the full metric suite (accuracy, vqa_soft_accuracy,
substring_match, token_f1, BLEU, METEOR, ROUGE-L, char_similarity,
llm_judge). Use --no_judge to skip the embedding-based LLM judge if you do
not have sentence-transformers installed.
"""

import argparse
import csv
import json
import os

from src.eval.metrics import compute_all_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved TextVQA predictions.")
    parser.add_argument("--input", type=str, required=True, help="Path to predictions.json.")
    parser.add_argument("--output", type=str, required=True, help="Path to save metrics JSON.")
    parser.add_argument(
        "--no_judge",
        action="store_true",
        help="Skip the LLM-as-a-Judge metric (faster, no extra dep).",
    )
    return parser.parse_args()


def load_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    preds = [row["prediction"] for row in rows]
    answers_list = [row["answers"] for row in rows]
    return preds, answers_list


def save_json(path, data):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(path, metrics):
    csv_path = path.replace(".json", ".csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def main():
    args = parse_args()

    preds, answers_list = load_predictions(args.input)
    judge_mode = "off" if args.no_judge else "auto"
    metrics = compute_all_metrics(preds, answers_list, judge=judge_mode)

    rounded = {k: round(v, 4) for k, v in metrics.items()}

    save_json(args.output, rounded)
    save_csv(args.output, rounded)

    print(json.dumps(rounded, indent=2))


if __name__ == "__main__":
    main()
