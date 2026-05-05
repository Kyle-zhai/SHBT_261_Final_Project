from datasets import load_dataset
import random


def load_textvqa(
    split="validation",
    sample_size=100,
    seed=42,
):
    """
    Load and sample TextVQA dataset.

    Returns:
        List[dict]:
            {
                "image": PIL.Image,
                "question": str,
                "answers": List[str]
            }
    """

    dataset = load_dataset("lmms-lab/textvqa", split=split)

    # ===== 随机抽样 =====
    random.seed(seed)
    if sample_size < len(dataset):
        indices = random.sample(range(len(dataset)), sample_size)
        dataset = dataset.select(indices)

    data = []
    for item in dataset:
        data.append({
            "image": item["image"],           # PIL Image
            "question": item["question"],
            "answers": item["answers"],       # 保留所有答案
        })

    return data