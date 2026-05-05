from __future__ import annotations

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class BLIP2Model:
    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device: str | None = None,
        max_new_tokens: int = 20,
    ):
        self.model_name = model_name
        self.device = device or self._get_default_device()
        self.max_new_tokens = max_new_tokens

        self.processor = Blip2Processor.from_pretrained(model_name)

        dtype = self._get_dtype()
        self.model = Blip2ForConditionalGeneration.from_pretrained(
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
            return torch.float16
        return torch.float32

    def generate(self, image: Image.Image, prompt: str) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        output = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        return self._postprocess(output, prompt)

    @staticmethod
    def _postprocess(output: str, prompt: str) -> str:
        text = output.strip()

        # 有些情况下模型会把prompt一起回出来，简单去掉
        if text.startswith(prompt):
            text = text[len(prompt):].strip()

        # 只保留第一行，减少跑偏
        text = text.split("\n")[0].strip()
        return text