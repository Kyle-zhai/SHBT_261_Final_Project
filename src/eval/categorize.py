"""Question-type categorization for TextVQA error analysis.

We assign each (question, answers) pair a single category by applying ordered
rules. The rules are based on the surface form of the question because TextVQA
does not ship official categories.

Categories:
    yes_no    -> binary questions ("is/are/does/can ...")
    time      -> clock / time questions
    price     -> currency / cost questions
    date_year -> calendar year questions ("what year ...")
    number    -> numeric questions that are not date/price/time
    color     -> color questions
    brand     -> brand / company / make questions
    text_read -> generic "what does it say / what is written" type questions
    other     -> fallback
"""

import re
from collections import defaultdict
from typing import Dict, List


_CATEGORY_RULES = [
    ("yes_no", [
        r"^\s*(is|are|do|does|did|can|could|will|would|has|have|was|were)\b",
    ]),
    ("time", [
        r"\bwhat\s+time\b",
        r"\btime\s+(does|is|on)\b",
        r"\bclock\b",
    ]),
    ("price", [
        r"\bhow\s+much\b",
        r"\bprice\b",
        r"\bcost\b",
        r"\$",
    ]),
    ("date_year", [
        r"\bwhat\s+year\b",
        r"\bwhich\s+year\b",
        r"\byear\s+(is|was)\b",
    ]),
    ("number", [
        r"\bhow\s+many\b",
        r"\bwhat\s+number\b",
        r"\bwhich\s+number\b",
        r"\bnumber\s+(is|of|on)\b",
        r"\bdigit",
        r"\bjersey\b",
    ]),
    ("color", [
        r"\bwhat\s+colou?r\b",
        r"\bwhich\s+colou?r\b",
    ]),
    ("brand", [
        r"\bbrand\b",
        r"\bcompany\b",
        r"\bmake\s+of\b",
        r"\bmaker\b",
        r"\bmanufacturer\b",
    ]),
    ("text_read", [
        r"\bwhat\s+(does|is)\s+.*\b(say|written|printed|read)",
        r"\bwhat\s+word\b",
        r"\bwhat\s+letter\b",
        r"\bwhat\s+text\b",
        r"\bname\s+of\b",
        r"\btitle\b",
        r"\blabel\b",
        r"\bsign\b",
    ]),
]


def categorize_question(question: str) -> str:
    q = (question or "").lower().strip()
    for category, patterns in _CATEGORY_RULES:
        for pat in patterns:
            if re.search(pat, q):
                return category
    return "other"


def per_category_breakdown(
    rows: List[Dict],
    metric_fn,
) -> Dict[str, Dict]:
    """Group rows by category and aggregate a per-row metric.

    Args:
        rows: list of {"question": str, "prediction": str, "answers": list[str]}.
        metric_fn: callable(pred, answers) -> float. Returned per-row score is
            averaged within each category.

    Returns:
        dict mapping category -> {"count": n, "score": mean}.
    """
    buckets = defaultdict(list)

    for row in rows:
        cat = categorize_question(row["question"])
        score = metric_fn(row["prediction"], row["answers"])
        buckets[cat].append(score)

    out = {}
    for cat, scores in buckets.items():
        out[cat] = {
            "count": len(scores),
            "score": sum(scores) / len(scores) if scores else 0.0,
        }
    return out
