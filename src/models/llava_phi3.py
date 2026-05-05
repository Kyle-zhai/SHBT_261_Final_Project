from __future__ import annotations

import gc

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


class LLaVAModel:
    def __init__(
        self,
        model_name: str = "xtuner/llava-phi-3-mini-hf",
        device: str | None = None,
        max_new_tokens: int = 20,
    ):
        self.model_name = model_name
        self.device = device or self._get_default_device()
        self.max_new_tokens = max_new_tokens

        dtype = self._get_dtype()
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_name)

        if getattr(self.processor, "patch_size", None) is None:
            patch_size = getattr(getattr(self.model.config, "vision_config", None), "patch_size", None)
            if patch_size is None:
                patch_size = 14
            self.processor.patch_size = int(patch_size)

        if getattr(self.processor, "vision_feature_select_strategy", None) is None:
            strategy = getattr(self.model.config, "vision_feature_select_strategy", None)
            if strategy is None:
                strategy = "default"
            self.processor.vision_feature_select_strategy = str(strategy)

        if getattr(self.processor, "num_additional_image_tokens", None) is None:
            self.processor.num_additional_image_tokens = 1

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

    def _resize_image(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")

        max_side = 512
        w, h = image.size
        scale = min(max_side / max(w, h), 1.0)

        if scale < 1.0:
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

        return image

    def _build_prompt(self, prompt: str) -> str:
        return f"USER: <image>\n{prompt}\nASSISTANT:"

    def generate(self, image: Image.Image, prompt: str) -> str:
        image = self._resize_image(image)
        text = self._build_prompt(prompt)

        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_ids_trimmed = generated_ids[:, input_length:]

        output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        del inputs
        del generated_ids
        del generated_ids_trimmed
        gc.collect()

        if self.device == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        return self._postprocess(output)

    @staticmethod
    def _postprocess(output: str) -> str:
        text = output.strip()
        text = text.split("\n")[0].strip()
        return text