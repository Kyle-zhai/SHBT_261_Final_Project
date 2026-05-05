# TextVQA Prompt Engineering Project

## Overview

This project evaluates the performance of pretrained vision-language models on the TextVQA task in a zero-shot setting. We focus on prompt engineering to improve model performance without fine-tuning.

The main goal is to understand:

* How different prompt designs affect performance
* Whether reasoning-style prompts help or hurt
* What types of errors dominate in TextVQA

---

## Project Structure


.
├── scripts/
│ ├── run_pilot.py # Run pilot experiment (model comparison)
│ ├── run_main.py # Run main experiment (prompt comparison)
│ ├── evaluate_saved_predictions.py # Compute metrics from saved predictions
│ └── export_error_figures.py # Export qualitative examples as image panels
│
├── src/
│ ├── data/
│ │ └── load_data.py # Load TextVQA dataset
│ │
│ ├── models/
│ │ ├── blip2.py # BLIP-2 model wrapper
│ │ ├── qwen_vl.py # Qwen2.5-VL model wrapper
│ │ ├── llava.py # LLaVA model wrapper
│ │ └── llava15.py # LLaVA-1.5 model wrapper (main model)
│ │
│ ├── prompts/
│ │ └── prompt_builder.py # All prompt templates
│ │
│ └── eval/
│ └── metrics.py # Evaluation metrics (accuracy, BLEU, F1, etc.)
│
├── outputs/
│ ├── pilot/ # Pilot experiment results
│ ├── main/ # Main experiment results
│ └── main_30/ # 30-sample prompt development results
│
├── figures/
│ └── examples/ # Exported qualitative example images
│
└── README.md


---

## Setup

Install dependencies:


pip install torch transformers datasets pillow nltk rouge-score


Download NLTK resources:


python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"


---

## Dataset

We use the TextVQA dataset from Hugging Face.

The dataset is loaded automatically in:


src/data/load_data.py


---

## Models

We evaluate the following pretrained models:

* BLIP-2 (2.7B)
* Qwen2.5-VL (3B)
* LLaVA-1.5 (7B) ← main model

All models are used in a **zero-shot setting** (no fine-tuning).

---

## Experiments

### 1. Pilot Experiment (Model Selection)

Run:


PYTHONPATH=. python scripts/run_pilot.py
--device cpu
--models blip2 qwen llava15
--sample_size 15
--max_new_tokens 8
--save_predictions


Purpose:

* Compare models
* Select the best model for further experiments

---

### 2. Main Experiment (Prompt Engineering)

Run:


PYTHONPATH=. python scripts/run_main.py
--device cpu
--model llava15
--prompts constrained ocr_exact key_focus textvqa_final
--sample_size 30
--max_new_tokens 8
--output_dir outputs/main
--save_predictions


Purpose:

* Evaluate different prompt designs

Use a separate output directory to avoid overwriting previous results. For example, for 15-sample prompt development:


PYTHONPATH=. python scripts/run_main.py
--device cpu
--model llava15
--prompts baseline cot ocr_short constrained ocr_exact key_focus minimal_answer textvqa_final
--sample_size 15
--max_new_tokens 8
--output_dir outputs/main_15
--save_predictions


---

## Prompt Design

All prompts are defined in:


src/prompts/prompt_builder.py


We design prompts based on error analysis:

| Prompt        | Purpose                            |
| ------------- | ---------------------------------- |
| baseline      | Default prompt                     |
| cot           | Chain-of-thought reasoning         |
| ocr_short     | OCR-aware short answer             |
| constrained   | Short answer constraint            |
| ocr_exact     | Force exact text copying           |
| key_focus     | Guide attention to relevant region |
| minimal_answer | Very short word/number answer     |
| textvqa_final | Combined optimized prompt          |

---

## Evaluation

We compute multiple metrics:

* Accuracy (exact match)
* Substring match
* Token-level F1
* Precision / Recall
* BLEU
* METEOR
* ROUGE-L

Run evaluation:


PYTHONPATH=. python scripts/evaluate_saved_predictions.py
--input outputs/main/llava15/textvqa_final/predictions.json
--output outputs/main/llava15/textvqa_final/metrics.json


This script evaluates saved predictions only. It does **not** rerun the model, so it is useful for recomputing metrics without repeating slow inference.

---

## Qualitative Example Export

To export selected TextVQA examples as image panels for the report, run:


PYTHONPATH=. python scripts/export_error_figures.py
--predictions outputs/main/llava15/textvqa_final/predictions.json
--output_dir figures/examples
--sample_size 30
--seed 42
--indices 2 3 9 12


Purpose:

* Reload the same sampled TextVQA examples
* Match prediction indices with original images
* Save example panels for qualitative error analysis

The exported images will be saved to:


figures/examples/


Use the same `--sample_size` and `--seed` as the original experiment so the exported images match the prediction file.

---

## Key Findings

* LLaVA-1.5 outperforms BLIP-2 and Qwen in zero-shot TextVQA
* Prompt engineering improves accuracy from ~40% to ~43–47%
* Chain-of-thought prompting hurts performance
* OCR-related errors dominate
* Performance plateaus across optimized prompts

---

## Notes

* All experiments are conducted without fine-tuning
* Performance is limited by OCR capability of the model
* Larger sample sizes produce more stable results
* Use different `--output_dir` values to avoid overwriting previous experiment outputs

---