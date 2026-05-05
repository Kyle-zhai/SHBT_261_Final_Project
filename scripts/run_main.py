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
from src.eval.metrics import compute_all_metrics
from src.models import BLIP2Model, QwenVLModel
from src.models.llava15 import LLaVA15Model
from src.prompts import build_prompt, list_prompts


def parse_args():
    parser = argparse.ArgumentParser(description="Run main TextVQA prompt experiments.")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to use.")
    parser.add_argument("--sample_size", type=int, default=30, help="Number of examples to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--model",
        type=str,
        default="llava15",
        choices=["blip2", "qwen", "llava15"],
        help="Model to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device argument passed to the script.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=[
            "baseline",
            "constrained",
            "ocr_exact",
            "key_focus",
            "minimal_answer",
            "textvqa_final",
            "verbatim_ocr",
            "few_shot",
        ],
        choices=list_prompts(),
        help="Prompt variants to evaluate.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help=(
            "Maximum number of generated tokens. Increased from 8 -> 32 because "
            "max_new_tokens=8 was truncating multi-word answers like "
            "'writing new york' to 'New York' and 'sunliners' to 'A'."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/main",
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Whether to save per-example predictions.",
    )
    parser.add_argument(
        "--save_full_metrics",
        action="store_true",
        help=(
            "If set, write the full metric suite (VQA-soft, BLEU, ROUGE, "
            "LLM-judge, etc.) per prompt instead of just accuracy."
        ),
    )
    return parser.parse_args()


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


def get_model(model_name: str, device: str, max_new_tokens: int):
    if model_name == "blip2":
        return BLIP2Model(device=device, max_new_tokens=max_new_tokens)

    if model_name == "qwen":
        return QwenVLModel(device=device, max_new_tokens=max_new_tokens)

    if model_name == "llava15":
        return LLaVA15Model(device=device, max_new_tokens=max_new_tokens)

    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_prompt(
    model,
    model_name: str,
    prompt_name: str,
    data: List[Dict],
    save_predictions: bool = False,
    save_full_metrics: bool = False,
):
    preds = []
    answers_list = []
    questions = []
    prediction_rows = []

    total_start = time.time()

    for idx, item in enumerate(tqdm(data, desc=f"{model_name}-{prompt_name}", leave=False), start=1):
        image = item["image"]
        question = item["question"]
        answers = item["answers"]

        prompt = build_prompt(prompt_name, question)

        example_start = time.time()
        try:
            pred = model.generate(image, prompt)
            error_msg = ""
        except Exception as e:
            pred = ""
            error_msg = str(e)
        example_end = time.time()

        preds.append(pred)
        answers_list.append(answers)
        questions.append(question)

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

    metric_mode = "auto" if save_full_metrics else "off"
    full_metrics = compute_all_metrics(preds, answers_list, judge=metric_mode)
    full_metrics = {k: round(v, 4) for k, v in full_metrics.items()}

    total_time = total_end - total_start
    avg_time = total_time / len(data) if data else 0.0

    return {
        "model": model_name,
        "prompt": prompt_name,
        "metrics": full_metrics,
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

    print(f"\nLoading model: {args.model}")
    model = get_model(
        model_name=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )

    summary_rows = []

    for prompt_name in args.prompts:
        print(f"\nEvaluating prompt: {prompt_name}")

        prompt_output_dir = os.path.join(args.output_dir, args.model, prompt_name)
        ensure_dir(prompt_output_dir)

        result = evaluate_prompt(
            model=model,
            model_name=args.model,
            prompt_name=prompt_name,
            data=data,
            save_predictions=args.save_predictions,
            save_full_metrics=args.save_full_metrics,
        )

        accuracy = result["metrics"].get("accuracy", 0.0)
        vqa_soft = result["metrics"].get("vqa_soft_accuracy", 0.0)
        print(
            f"{args.model}-{prompt_name}: "
            f"accuracy={accuracy:.4f}, vqa_soft={vqa_soft:.4f}, "
            f"avg_time={result['avg_time_sec']:.4f}s"
        )

        summary_row = {
            "model": result["model"],
            "prompt": result["prompt"],
            "num_samples": result["num_samples"],
            "total_time_sec": result["total_time_sec"],
            "avg_time_sec": result["avg_time_sec"],
        }
        summary_row.update(result["metrics"])
        summary_rows.append(summary_row)

        save_json(
            os.path.join(prompt_output_dir, "metrics.json"),
            result["metrics"],
        )

        if args.save_predictions:
            save_json(
                os.path.join(prompt_output_dir, "predictions.json"),
                result["predictions"],
            )

    save_csv(os.path.join(args.output_dir, f"{args.model}_main_summary.csv"), summary_rows)
    save_json(os.path.join(args.output_dir, f"{args.model}_main_summary.json"), summary_rows)

    cleanup_model(model)

    print("\nMain experiment finished.")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
