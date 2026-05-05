import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


class LLaVA15Model:
    def __init__(
        self,
        model_name="llava-hf/llava-1.5-7b-hf",
        device=None,
        max_new_tokens=20,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        self.model.eval()

    def generate(self, image: Image.Image, prompt: str) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )

        input_len = inputs["input_ids"].shape[1]
        output_ids = output_ids[:, input_len:]

        output = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0]

        return output.strip().split("\n")[0]