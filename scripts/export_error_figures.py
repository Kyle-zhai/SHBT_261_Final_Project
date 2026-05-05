import argparse
import json
import os
import textwrap

from PIL import Image, ImageDraw, ImageFont

from src.data.load_data import load_textvqa


def parse_args():
    parser = argparse.ArgumentParser(description="Export TextVQA examples as figure panels.")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions.json.")
    parser.add_argument("--output_dir", type=str, default="figures/examples", help="Output directory.")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--sample_size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[2, 3, 9, 12],
        help="1-based indices from predictions.json to export.",
    )
    return parser.parse_args()


def load_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_answer_list(answers):
    if not answers:
        return ""
    return answers[0]


def resize_keep_aspect(image, max_width=520):
    w, h = image.size
    if w <= max_width:
        return image
    scale = max_width / w
    return image.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)


def get_font(size=18):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_wrapped_text(draw, text, xy, font, max_chars=58, line_spacing=6, fill=(0, 0, 0)):
    x, y = xy
    lines = []
    for part in text.split("\n"):
        lines.extend(textwrap.wrap(part, width=max_chars) or [""])

    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + line_spacing

    return y


def make_panel(image, row, output_path):
    image = image.convert("RGB")
    image = resize_keep_aspect(image, max_width=520)

    font_title = get_font(20)
    font_body = get_font(18)

    question = row["question"]
    prediction = row["prediction"]
    answer = normalize_answer_list(row["answers"])

    text_block_height = 170
    canvas_w = image.width
    canvas_h = image.height + text_block_height

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(image, (0, 0))

    draw = ImageDraw.Draw(canvas)
    y = image.height + 10

    y = draw_wrapped_text(draw, f"Q: {question}", (10, y), font_body)
    y = draw_wrapped_text(draw, f"Pred: {prediction}", (10, y + 2), font_body)
    y = draw_wrapped_text(draw, f"GT: {answer}", (10, y + 2), font_body)

    canvas.save(output_path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_textvqa(
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    preds = load_predictions(args.predictions)

    for idx in args.indices:
        data_idx = idx - 1
        if data_idx < 0 or data_idx >= len(data):
            print(f"Skipping invalid index: {idx}")
            continue

        image = data[data_idx]["image"]
        row = preds[data_idx]

        output_path = os.path.join(args.output_dir, f"example_{idx:02d}.png")
        make_panel(image, row, output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()