#!/bin/bash

# ============================================================
# Final Project: Visual Understanding with TextVQA
# AI in Medicine - Spring 2026
# Brown University - SLURM job script (Oscar)
#
# Usage:
#   sbatch slurm_run.sh                   # run all stages (pilot + main + analyze)
#   sbatch slurm_run.sh pilot             # just pilot (model selection)
#   sbatch slurm_run.sh main              # just main (prompt comparison)
#   sbatch slurm_run.sh analyze           # just per-category analysis on saved preds
#
# Monitor your job:
#   myq                       # check job status
#   cat slurm-<jobid>.out     # view stdout
#   cat slurm-<jobid>.err     # view stderr
# ============================================================

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -J textvqa
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

STAGE=${1:-all}

PILOT_SAMPLES=${PILOT_SAMPLES:-100}
MAIN_SAMPLES=${MAIN_SAMPLES:-200}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
SEED=${SEED:-42}

PILOT_MODELS="blip2 qwen llava15"
MAIN_PROMPTS="baseline cot ocr_short constrained ocr_exact key_focus minimal_answer textvqa_final verbatim_ocr few_shot"

PILOT_DIR=outputs/pilot
MAIN_DIR=outputs/main_oscar

echo "============================================"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Stage:     $STAGE"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "PilotSamp: $PILOT_SAMPLES"
echo "MainSamp:  $MAIN_SAMPLES"
echo "MaxTokens: $MAX_NEW_TOKENS"
echo "============================================"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Activate the project-local venv (created with `python -m venv .venv` or
# `uv venv`). We use plain python instead of `uv run` because Oscar may have a
# parent pyproject (e.g. from another course) that uv prefers over our venv.
if [ -d ".venv" ]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
fi

PY="python"
echo "Installing deps into active env..."
pip install -q -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Avoid OpenMP duplicate-runtime crash if it shows up.
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=.

# Make sure NLTK resources METEOR needs are present.
$PY -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True); nltk.download('punkt_tab', quiet=True)"

run_pilot() {
    echo
    echo "===== STAGE: pilot (model selection) ====="
    $PY scripts/run_pilot.py \
        --device cuda \
        --models $PILOT_MODELS \
        --sample_size "$PILOT_SAMPLES" \
        --seed "$SEED" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --output_dir "$PILOT_DIR" \
        --save_predictions

    for m in $PILOT_MODELS; do
        if [ -f "$PILOT_DIR/$m/predictions.json" ]; then
            $PY scripts/evaluate_saved_predictions.py \
                --input "$PILOT_DIR/$m/predictions.json" \
                --output "$PILOT_DIR/$m/metrics.json"
        fi
    done
}

run_main() {
    echo
    echo "===== STAGE: main (prompt engineering) ====="
    $PY scripts/run_main.py \
        --device cuda \
        --model llava15 \
        --prompts $MAIN_PROMPTS \
        --sample_size "$MAIN_SAMPLES" \
        --seed "$SEED" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --output_dir "$MAIN_DIR" \
        --save_predictions \
        --save_full_metrics
}

run_analyze() {
    echo
    echo "===== STAGE: analyze (per-category breakdown) ====="
    for p in $MAIN_PROMPTS; do
        pred_file="$MAIN_DIR/llava15/$p/predictions.json"
        if [ -f "$pred_file" ]; then
            echo
            echo "--- prompt=$p ---"
            $PY scripts/analyze_categories.py \
                --input "$pred_file" \
                --output_dir "$MAIN_DIR/llava15/$p" \
                --metrics accuracy vqa_soft substring char_sim llm_judge
        else
            echo "skip $p (no predictions.json)"
        fi
    done
}

case "$STAGE" in
    pilot)   run_pilot ;;
    main)    run_main ;;
    analyze) run_analyze ;;
    all)     run_pilot; run_main; run_analyze ;;
    *)       echo "unknown stage: $STAGE"; exit 1 ;;
esac

echo
echo "============================================"
echo "Finished:  $(date)"
echo "Outputs:"
echo "  pilot  -> $PILOT_DIR"
echo "  main   -> $MAIN_DIR"
echo "============================================"
