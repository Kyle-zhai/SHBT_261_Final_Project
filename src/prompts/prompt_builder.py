"""Prompt templates for TextVQA experiments.

Each function takes the raw question string and returns the full text prompt
that will be paired with the image and fed into the VLM. Prompts are grouped
in PROMPT_REGISTRY so scripts can iterate over them by name.

Design notes (informed by error analysis on val/main_30):
    * Many wrong answers were short single-token outputs because the original
      runs used max_new_tokens=8 and the model truncated multi-word answers
      such as "writing new york" -> "New York". The new prompts therefore
      reference multi-word answers explicitly and the calling scripts now use
      max_new_tokens >= 24.
    * Chain-of-thought prompting hurt accuracy (13% on val/main, vs 40% for
      baseline). It is kept here only for completeness / ablation.
    * Two new prompts target the dominant error modes:
        - `few_shot`        : in-context examples to anchor short copy-style
                              answers (clocks, prices, brand names).
        - `verbatim_ocr`    : explicit "copy the visible text verbatim" with
                              format hints for time / price / number.
"""

from typing import Callable, Dict


def _baseline(question: str) -> str:
    return (
        "Answer the question based on the image.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _cot(question: str) -> str:
    return (
        "Think step by step about what the image shows and what text is "
        "visible. Then give the final short answer.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _ocr_short(question: str) -> str:
    return (
        "Read the text in the image carefully and answer the question.\n"
        "Return only the final answer.\n"
        "Do not explain.\n"
        "Use the shortest possible phrase.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _constrained(question: str) -> str:
    return (
        "Answer using only the exact word or phrase from the image when possible.\n"
        "Do not explain.\n"
        "Do not use a full sentence.\n"
        "If the answer is a number or time, output only that value.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _ocr_exact(question: str) -> str:
    return (
        "The answer is a word or phrase written in the image.\n"
        "Read the text carefully.\n"
        "Copy the answer exactly as it appears.\n"
        "Do not change spelling.\n"
        "Do not explain.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _key_focus(question: str) -> str:
    return (
        "Focus only on the part of the image relevant to the question.\n"
        "Ignore unrelated objects.\n"
        "Find the key text or label and answer with a short phrase.\n"
        "No explanation.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _minimal_answer(question: str) -> str:
    return (
        "Answer with only the final word, number, or short phrase.\n"
        "No explanation.\n"
        "No extra sentence.\n"
        "No punctuation unless it is part of a time, price, or brand name.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _textvqa_final(question: str) -> str:
    return (
        "Answer the question by reading the visible text in the image.\n"
        "Focus only on the region relevant to the question.\n"
        "Copy the exact word, number, time, price, brand, or phrase from the image.\n"
        "Do not explain.\n"
        "Do not answer with a full sentence.\n"
        "If the answer contains multiple words, include the complete phrase.\n"
        "If the answer is a time, price, or number, output only that value.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _verbatim_ocr(question: str) -> str:
    """Optimized prompt v2.

    Targets the most common error patterns:
      * single-word truncation of multi-word answers
      * dropping currency / unit symbols ($, %, °C)
      * substituting brand names that look similar
    """
    return (
        "You are reading text printed inside the image.\n"
        "Steps:\n"
        "1. Locate the region the question asks about.\n"
        "2. Copy the exact text you see there, character by character.\n"
        "3. Keep currency symbols, punctuation, and digits exactly as written.\n"
        "4. Include all words that belong together (e.g. 'writing new york', 'castrol edge').\n"
        "5. Do not paraphrase, translate, or guess.\n"
        "6. Do not output a sentence. Output only the answer.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _few_shot(question: str) -> str:
    """In-context examples covering the four hardest TextVQA categories.

    Examples are short and demonstrate the desired output style:
      time -> HH:MM, price -> $X.XX, brand -> exact spelling, multi-word phrase
    kept whole.
    """
    return (
        "Read the text visible in the image and answer with the shortest exact "
        "phrase from the image. Match the format of these examples.\n\n"
        "Question: what time does the clock show?\n"
        "Answer: 10:07\n\n"
        "Question: how much is the headband?\n"
        "Answer: $2.00\n\n"
        "Question: what brand is the bottle?\n"
        "Answer: southern comfort\n\n"
        "Question: what is the title of this book?\n"
        "Answer: writing new york\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


PROMPT_REGISTRY: Dict[str, Callable[[str], str]] = {
    "baseline": _baseline,
    "cot": _cot,
    "ocr_short": _ocr_short,
    "constrained": _constrained,
    "ocr_exact": _ocr_exact,
    "key_focus": _key_focus,
    "minimal_answer": _minimal_answer,
    "textvqa_final": _textvqa_final,
    "verbatim_ocr": _verbatim_ocr,
    "few_shot": _few_shot,
}


def build_prompt(prompt_name: str, question: str) -> str:
    """Return the rendered prompt string for the requested template."""
    if prompt_name not in PROMPT_REGISTRY:
        raise ValueError(
            f"Unsupported prompt: {prompt_name}. "
            f"Available: {sorted(PROMPT_REGISTRY)}"
        )
    return PROMPT_REGISTRY[prompt_name](question)


def list_prompts():
    """List all registered prompt names."""
    return sorted(PROMPT_REGISTRY)
