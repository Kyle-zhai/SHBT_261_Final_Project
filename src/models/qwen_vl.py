from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class QwenVLModel:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str | None = None,
        max_new_tokens: int = 20,
    ):
        self.model_name = model_name
        self.device = device or self._get_default_device()
        self.max_new_tokens = max_new_tokens

        self.processor = AutoProcessor.from_pretrained(model_name)

        dtype = self._get_dtype()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(self.device)

        self.model.eval()

    def _get_default_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_dtype(self):
        if self.device == "cuda":
            return torch.bfloat16
        return torch.float32

    def _build_messages(self, prompt: str):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _resize_image(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")

        max_side = 512
        w, h = image.size
        scale = min(max_side / max(w, h), 1.0)

        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

        return image

    def generate(self, image: Image.Image, prompt: str) -> str:
        image = self._resize_image(image)

        messages = self._build_messages(prompt)

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return self._postprocess(output)

    @staticmethod
    def _postprocess(output: str) -> str:
        text = output.strip()
        text = text.split("\n")[0].strip()
        return text