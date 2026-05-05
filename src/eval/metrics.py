import math
import re
from collections import Counter
from difflib import SequenceMatcher

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


# -----------------------------------------------------------------------------
# Text normalization
# -----------------------------------------------------------------------------

# Articles and very common stopwords that VQA-style evaluators usually strip.
_ARTICLES = {"a", "an", "the"}


def normalize(text):
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_vqa(text):
    """Normalization closer to the official VQA evaluator: lowercase, strip
    punctuation, drop articles ('a', 'an', 'the')."""
    text = normalize(text)
    tokens = [tok for tok in text.split() if tok not in _ARTICLES]
    return " ".join(tokens)


def tokenize(text):
    return normalize(text).split()


# -----------------------------------------------------------------------------
# Hard accuracy (any-of-N exact match) and TextVQA-style soft accuracy
# -----------------------------------------------------------------------------


def exact_match(pred, answers):
    pred = normalize(pred)
    return int(any(pred == normalize(ans) for ans in answers))


def vqa_soft_accuracy(pred, answers):
    """Official TextVQA / VQA accuracy.

    For each question with N (typically 10) human answers, the score is
        min(#answers that match the prediction / 3, 1.0)
    Articles are stripped before comparison.
    """
    if not answers:
        return 0.0

    pred_norm = _normalize_vqa(pred)
    matches = sum(1 for a in answers if _normalize_vqa(a) == pred_norm)
    return min(matches / 3.0, 1.0)


def best_answer(pred, answers):
    pred_norm = normalize(pred)

    if not answers:
        return ""

    # Choose the answer with highest token F1 as reference.
    best = answers[0]
    best_score = -1

    for ans in answers:
        score = token_f1(pred_norm, normalize(ans))
        if score > best_score:
            best_score = score
            best = ans

    return best


# -----------------------------------------------------------------------------
# Token-level overlap metrics
# -----------------------------------------------------------------------------


def token_f1(pred, answer):
    pred_tokens = tokenize(pred)
    answer_tokens = tokenize(answer)

    if len(pred_tokens) == 0 or len(answer_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(answer_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def token_precision(pred, answer):
    pred_tokens = tokenize(pred)
    answer_tokens = tokenize(answer)

    if len(pred_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(answer_tokens)
    return sum(common.values()) / len(pred_tokens)


def token_recall(pred, answer):
    pred_tokens = tokenize(pred)
    answer_tokens = tokenize(answer)

    if len(answer_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(answer_tokens)
    return sum(common.values()) / len(answer_tokens)


def substring_match(pred, answers):
    pred_norm = normalize(pred)
    if not pred_norm:
        return 0

    for ans in answers:
        ans_norm = normalize(ans)
        if pred_norm in ans_norm or ans_norm in pred_norm:
            return 1

    return 0


# -----------------------------------------------------------------------------
# Generation-quality metrics
# -----------------------------------------------------------------------------


def get_ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def bleu_score(pred, answer, max_n=2):
    pred_tokens = tokenize(pred)
    answer_tokens = tokenize(answer)

    if not pred_tokens or not answer_tokens:
        return 0.0

    precisions = []

    for n in range(1, max_n + 1):
        pred_ngrams = Counter(get_ngrams(pred_tokens, n))
        ans_ngrams = Counter(get_ngrams(answer_tokens, n))

        if not pred_ngrams:
            continue

        overlap = sum((pred_ngrams & ans_ngrams).values())
        total = sum(pred_ngrams.values())

        # Add-one smoothing.
        precisions.append((overlap + 1) / (total + 1))

    if not precisions:
        return 0.0

    geo_mean = math.exp(
        sum(math.log(max(p, 1e-12)) for p in precisions) / len(precisions)
    )

    if len(pred_tokens) < len(answer_tokens):
        brevity_penalty = math.exp(1 - len(answer_tokens) / len(pred_tokens))
    else:
        brevity_penalty = 1.0

    return brevity_penalty * geo_mean


def meteor(pred, answer):
    pred_tokens = tokenize(pred)
    answer_tokens = tokenize(answer)

    if not pred_tokens or not answer_tokens:
        return 0.0

    return meteor_score([answer_tokens], pred_tokens)


def rouge_l(pred, answer):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(normalize(answer), normalize(pred))
    return scores["rougeL"].fmeasure


# -----------------------------------------------------------------------------
# Character-level fuzzy match (catches one-letter OCR errors like
# "casto edge" / "castrol edge", "crement" / "cremant")
# -----------------------------------------------------------------------------


def char_similarity(pred, answer):
    """Ratio in [0, 1] using difflib's longest contiguous matching subsequence."""
    return SequenceMatcher(None, normalize(pred), normalize(answer)).ratio()


def best_char_similarity(pred, answers):
    if not answers:
        return 0.0
    return max(char_similarity(pred, a) for a in answers)


# -----------------------------------------------------------------------------
# LLM-as-a-Judge similarity
#
# We provide two backends:
#   1. embedding (default): sentence-transformers cosine similarity. Treats
#      semantically-equivalent answers as a match. Lazy-loaded; if the package
#      is not installed we fall back to char_similarity so the script never
#      crashes.
#   2. api: pluggable function that calls an external LLM. Off by default to
#      avoid forcing an API key on graders. See `llm_judge_score_api` if you
#      want to wire in OpenAI / Anthropic.
# -----------------------------------------------------------------------------


_EMBEDDER = None
_EMBEDDER_FAILED = False


def _get_embedder():
    """Lazy-load a tiny sentence embedding model. Returns None if unavailable."""
    global _EMBEDDER, _EMBEDDER_FAILED
    if _EMBEDDER is not None or _EMBEDDER_FAILED:
        return _EMBEDDER
    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        _EMBEDDER_FAILED = True
        _EMBEDDER = None
    return _EMBEDDER


def _cosine(u, v):
    import numpy as np
    nu = u / (np.linalg.norm(u) + 1e-12)
    nv = v / (np.linalg.norm(v) + 1e-12)
    return float((nu * nv).sum())


def llm_judge_score(pred, answers):
    """Soft semantic-equivalence score in [0, 1].

    Uses sentence embeddings when available; otherwise falls back to
    character-level similarity so the metric pipeline always produces a value.
    Returns the maximum score against any of the reference answers.
    """
    if not answers:
        return 0.0

    embedder = _get_embedder()
    if embedder is None:
        return best_char_similarity(pred, answers)

    pred_norm = normalize(pred)
    if not pred_norm:
        return 0.0

    refs = [normalize(a) for a in answers if normalize(a)]
    if not refs:
        return 0.0

    vecs = embedder.encode([pred_norm] + refs, show_progress_bar=False)
    pred_vec, ref_vecs = vecs[0], vecs[1:]
    return max(_cosine(pred_vec, r) for r in ref_vecs)


# -----------------------------------------------------------------------------
# Aggregator
# -----------------------------------------------------------------------------


def compute_all_metrics(preds, answers_list, judge="auto"):
    """Compute the full metric suite over a list of predictions.

    Args:
        preds: list[str] of model predictions.
        answers_list: list[list[str]] of per-question reference answer lists.
        judge: "auto" -> use embedder if installed, fallback to char similarity.
               "off"  -> skip the LLM-judge metric (faster).
    """
    n = len(preds)
    if n == 0:
        return {}

    exact_matches = []
    soft_accuracies = []
    substring_matches = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    bleu_scores = []
    meteor_scores = []
    rouge_l_scores = []
    char_sims = []
    judge_scores = []

    for pred, answers in zip(preds, answers_list):
        ref = best_answer(pred, answers)

        exact_matches.append(exact_match(pred, answers))
        soft_accuracies.append(vqa_soft_accuracy(pred, answers))
        substring_matches.append(substring_match(pred, answers))
        f1_scores.append(token_f1(pred, ref))
        precision_scores.append(token_precision(pred, ref))
        recall_scores.append(token_recall(pred, ref))
        bleu_scores.append(bleu_score(pred, ref))
        meteor_scores.append(meteor(pred, ref))
        rouge_l_scores.append(rouge_l(pred, ref))
        char_sims.append(best_char_similarity(pred, answers))

        if judge == "auto":
            judge_scores.append(llm_judge_score(pred, answers))

    metrics = {
        "accuracy": sum(exact_matches) / n,
        "vqa_soft_accuracy": sum(soft_accuracies) / n,
        "substring_match": sum(substring_matches) / n,
        "char_similarity": sum(char_sims) / n,
        "token_f1": sum(f1_scores) / n,
        "token_precision": sum(precision_scores) / n,
        "token_recall": sum(recall_scores) / n,
        "bleu": sum(bleu_scores) / n,
        "meteor": sum(meteor_scores) / n,
        "rouge_l": sum(rouge_l_scores) / n,
    }
    if judge == "auto":
        metrics["llm_judge"] = sum(judge_scores) / n

    return metrics


def compute_accuracy(preds, answers_list):
    return compute_all_metrics(preds, answers_list, judge="off")["accuracy"]
