from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from typing import Dict, List

import torch
from tqdm import tqdm

from src.data.load_data import load_textvqa
from src.eval.metrics import compute_accuracy
from src.models import BLIP2Model, LLaVAModel, QwenVLModel
from src.models.llava15 import LLaVA15Model


def parse_args():
    parser = argparse.ArgumentParser(description="Run pilot experiments for TextVQA model selection.")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to use.")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of examples to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["blip2", "llava", "qwen", "llava15"],
        choices=["blip2", "llava", "qwen", "llava15"],
        help="Models to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=[None, "cpu", "mps", "cuda"],
        help="Device to use. If not set, it will be auto-detected.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Maximum generation length.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/pilot",
        help="Directory to save pilot outputs.",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Whether to save per-example predictions for each model.",
    )
    return parser.parse_args()


def build_prompt(question: str) -> str:
    return f"Answer the question based on the image.\n\nQuestion: {question}\nAnswer:"


def get_model(model_name: str, device: str | None, max_new_tokens: int):
    if model_name == "blip2":
        return BLIP2Model(device=device, max_new_tokens=max_new_tokens)
    if model_name == "llava":
        return LLaVAModel(device=device, max_new_tokens=max_new_tokens)
    if model_name == "qwen":
        return QwenVLModel(device=device, max_new_tokens=max_new_tokens)
    if model_name == "llava15":
        return LLaVA15Model(device=device, max_new_tokens=max_new_tokens)
    raise ValueError(f"Unsupported model: {model_name}")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(path: str, rows: List[Dict]):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cleanup_model(model):
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def evaluate_model(model_name: str, model, data: List[Dict], save_predictions: bool = False):
    preds = []
    answers_list = []
    prediction_rows = []

    total_start = time.time()

    for idx, item in enumerate(tqdm(data, desc=f"Running {model_name}", leave=False), start=1):
        image = item["image"]
        question = item["question"]
        answers = item["answers"]

        prompt = build_prompt(question)

        example_start = time.time()
        try:
            pred = model.generate(image, prompt)
        except Exception as e:
            pred = ""
            error_msg = str(e)
        else:
            error_msg = ""
        example_end = time.time()

        preds.append(pred)
        answers_list.append(answers)

        if save_predictions:
            prediction_rows.append(
                {
                    "index": idx,
                    "question": question,
                    "prediction": pred,
                    "answers": answers,
                    "latency_sec": round(example_end - example_start, 4),
                    "error": error_msg,
                }
            )

    total_end = time.time()

    accuracy = compute_accuracy(preds, answers_list)
    total_time = total_end - total_start
    avg_time = total_time / len(data) if data else 0.0

    return {
        "model": model_name,
        "accuracy": round(accuracy, 4),
        "num_samples": len(data),
        "total_time_sec": round(total_time, 4),
        "avg_time_sec": round(avg_time, 4),
        "predictions": prediction_rows,
    }


def main():
    args = parse_args()

    ensure_dir(args.output_dir)

    print("Loading dataset...")
    data = load_textvqa(
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(f"Loaded {len(data)} examples from split='{args.split}'.")

    sampled_metadata = []
    for idx, item in enumerate(data, start=1):
        sampled_metadata.append(
            {
                "index": idx,
                "question": item["question"],
                "answers": item["answers"],
            }
        )

    save_json(os.path.join(args.output_dir, "sampled_examples.json"), sampled_metadata)

    summary_rows = []

    for model_name in args.models:
        print(f"\nEvaluating model: {model_name}")
        model_output_dir = os.path.join(args.output_dir, model_name)
        ensure_dir(model_output_dir)

        model = get_model(
            model_name=model_name,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )

        result = evaluate_model(
            model_name=model_name,
            model=model,
            data=data,
            save_predictions=args.save_predictions,
        )

        print(
            f"{model_name}: "
            f"accuracy={result['accuracy']:.4f}, "
            f"avg_time={result['avg_time_sec']:.4f}s"
        )

        summary_rows.append(
            {
                "model": result["model"],
                "accuracy": result["accuracy"],
                "num_samples": result["num_samples"],
                "total_time_sec": result["total_time_sec"],
                "avg_time_sec": result["avg_time_sec"],
            }
        )

        if args.save_predictions:
            save_json(
                os.path.join(model_output_dir, "predictions.json"),
                result["predictions"],
            )

        cleanup_model(model)

    save_csv(os.path.join(args.output_dir, "pilot_summary.csv"), summary_rows)
    save_json(os.path.join(args.output_dir, "pilot_summary.json"), summary_rows)

    print("\nPilot experiment finished.")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()