"""
TrOCR (Vision Encoder Decoder) baseline wrapper.
Supports dual-path dynamic domain routing:
  - 'microsoft/trocr-base-printed' for standard & historical printed documents
  - 'microsoft/trocr-base-handwritten' for cursive & handwritten scripts
"""

import os
import numpy as np
import torch
from PIL import Image

# Configure optimal intra-op parallelism for fast CPU inference
if not torch.cuda.is_available():
    try:
        n_threads = min(8, max(2, os.cpu_count() or 4))
        torch.set_num_threads(n_threads)
    except Exception:
        pass

PRINTED = "microsoft/trocr-base-printed"
HANDWRITTEN = "microsoft/trocr-base-handwritten"


class TrOCRBaseline:
    """
    Dual-Path Baseline Vision Transformer OCR wrapper.
    Automatically routes between printed and handwritten checkpoints.
    Optimized for high accuracy and fast CPU/GPU inference.
    """

    def __init__(self, model_name: str = PRINTED, strict: bool = False,
                 max_new_tokens: int = 36, num_beams: int = 1):
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.hw_processor = None
        self.hw_model = None
        self.loaded = False
        self.hw_loaded = False
        self.n_params = 0
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model(strict=strict)

    def _load_model(self, strict: bool):
        import warnings
        warnings.filterwarnings("ignore")
        try:
            import transformers
            transformers.logging.set_verbosity_error()
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel, RobertaTokenizer, ViTImageProcessor

            # Try loading directly from local cache first for sub-second startup
            try:
                tokenizer = RobertaTokenizer.from_pretrained(self.model_name, local_files_only=True)
                image_processor = ViTImageProcessor.from_pretrained(self.model_name, local_files_only=True)
                self.processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
            except Exception:
                try:
                    self.processor = TrOCRProcessor.from_pretrained(self.model_name)
                except Exception:
                    tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
                    image_processor = ViTImageProcessor.from_pretrained(self.model_name)
                    self.processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

            try:
                self.model = VisionEncoderDecoderModel.from_pretrained(
                    self.model_name, local_files_only=True
                ).to(self.device)
            except Exception:
                self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device)

            self.model.eval()
            self.n_params = sum(p.numel() for p in self.model.parameters())
            self.loaded = True
            print(f"[*] TrOCR loaded: {self.model_name} "
                  f"({self.n_params / 1e6:.1f}M parameters) on {self.device}")
        except Exception as exc:
            msg = f"Could not load TrOCR '{self.model_name}': {exc}"
            if strict:
                raise RuntimeError(msg) from exc
            print(f"[!] {msg}")
            self.loaded = False

    def _get_handwritten_model(self):
        """Lazy loads the handwritten model on demand."""
        if self.hw_loaded and self.hw_model is not None:
            return self.hw_model, self.hw_processor

        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel, RobertaTokenizer, ViTImageProcessor
            print(f"[*] Lazy-loading handwritten TrOCR ({HANDWRITTEN})...")
            try:
                tokenizer = RobertaTokenizer.from_pretrained(HANDWRITTEN, local_files_only=True)
                image_processor = ViTImageProcessor.from_pretrained(HANDWRITTEN, local_files_only=True)
                self.hw_processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
            except Exception:
                try:
                    self.hw_processor = TrOCRProcessor.from_pretrained(HANDWRITTEN)
                except Exception:
                    tokenizer = RobertaTokenizer.from_pretrained(HANDWRITTEN)
                    image_processor = ViTImageProcessor.from_pretrained(HANDWRITTEN)
                    self.hw_processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

            try:
                self.hw_model = VisionEncoderDecoderModel.from_pretrained(
                    HANDWRITTEN, local_files_only=True
                ).to(self.device)
            except Exception:
                self.hw_model = VisionEncoderDecoderModel.from_pretrained(HANDWRITTEN).to(self.device)

            self.hw_model.eval()
            self.hw_loaded = True
            print(f"[OK] Handwritten TrOCR ({HANDWRITTEN}) ready.")
            return self.hw_model, self.hw_processor
        except Exception as exc:
            print(f"[!] Failed to load handwritten TrOCR ({exc}); falling back to primary model.")
            return self.model, self.processor

    @staticmethod
    def _sanitize_output(text: str) -> str:
        """
        Claude-style text cleanup: strips optical noise, random hash symbols,
        repeated punctuation, and trailing noise tokens.
        """
        if not text:
            return ""
        import re
        t = text.strip()
        # Strip leading/trailing hashes, tildes, backticks, bullets, stray quotes
        t = re.sub(r'^[#~*•|`"\'\^]+\s*', '', t)
        t = re.sub(r'\s*[#~*•|`"\'\^]+$', '', t)
        # Strip trailing stray tokens like '# O.E', '# 1 .', or stray trailing hashes
        t = re.sub(r'\s*#\s*[A-Za-z0-9.]+$', '', t)
        t = re.sub(r'\s*#+$', '', t)
        # Collapse multiple spaces
        t = re.sub(r'\s{2,}', ' ', t)
        return t.strip()

    @staticmethod
    def _to_pil(image_input):
        import cv2
        if isinstance(image_input, np.ndarray):
            img_np = image_input.copy()
            if img_np.ndim == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[2] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        else:
            img_np = np.array(image_input.convert("RGB"))

        h, w = img_np.shape[:2]
        if h <= 0 or w <= 0:
            blank = np.ones((384, 384, 3), dtype=np.uint8) * 255
            return Image.fromarray(blank), blank

        # 1. Background Polarity Check (Dark Mode / Chalkboard Inversion)
        # If image has dark background with light text, invert to standard dark-on-light
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        if float(np.median(border)) < 115:
            img_np = 255 - img_np

        # 2. Add clean margin padding so characters never clip against edges
        pad = 12
        img_np = cv2.copyMakeBorder(
            img_np, pad, pad, pad, pad,
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )

        pil = Image.fromarray(img_np)
        return pil, img_np

    def predict(self, image_input, domain: str = "auto") -> dict:
        """
        Generate text for one image with dynamic domain routing.
        domain: 'auto' | 'printed' | 'handwritten' | 'math'
        """
        image_pil, img_np = self._to_pil(image_input)

        from src.preprocessing.enhancement import ImagePreprocessor
        if domain == "auto":
            detected_domain = ImagePreprocessor.detect_domain(img_np)
        else:
            detected_domain = domain.lower()

        active_model = self.model
        active_proc = self.processor
        model_tag = self.model_name

        if detected_domain == "handwritten" and self.model_name == PRINTED:
            active_model, active_proc = self._get_handwritten_model()
            model_tag = f"{self.model_name} -> {HANDWRITTEN}"

        text = ""
        confidence = None

        if self.loaded and active_model is not None:
            try:
                pixel_values = active_proc(
                    images=image_pil, return_tensors="pt"
                ).pixel_values.to(self.device)

                with torch.inference_mode():
                    generate_kwargs = {
                        "max_new_tokens": self.max_new_tokens,
                        "num_beams": self.num_beams,
                        "use_cache": True,
                        "no_repeat_ngram_size": 3,
                        "output_scores": True,
                        "return_dict_in_generate": True,
                    }
                    if self.num_beams > 1:
                        generate_kwargs["early_stopping"] = True
                    out = active_model.generate(pixel_values, **generate_kwargs)
                ids = out.sequences
                raw_decoded = active_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
                text = self._sanitize_output(raw_decoded)

                score = getattr(out, "sequences_scores", None)
                if score is not None and len(score) > 0:
                    confidence = float(min(1.0, max(0.0, torch.exp(score[0]).item())))
                elif getattr(out, "scores", None):
                    # Average token probability for greedy decoding
                    try:
                        token_probs = [torch.softmax(s[0], dim=-1).max().item() for s in out.scores]
                        confidence = float(np.mean(token_probs)) if token_probs else 0.9
                    except Exception:
                        confidence = 0.9
                else:
                    confidence = 0.9
            except Exception as exc:
                print(f"[!] TrOCR generation failed: {exc}")
                text = ""

        from src.utils.math_recognizer import MathFormulaParser
        from src.utils.postprocessing import OCRPostProcessor

        is_math = bool(
            detected_domain == "math"
            or MathFormulaParser.has_math_visual_structure(img_np)
            or OCRPostProcessor.is_math_expression(text)
        )

        return {
            "model": f"Vision Transformer ({model_tag})",
            "predicted_text": text,
            "is_math": is_math,
            "detected_domain": detected_domain,
            "confidence": confidence,
            "parameters_count": int(self.n_params),
            "loaded": self.loaded,
            "note": "Raw TrOCR generation with domain routing.",
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Smoke-test the TrOCR baseline")
    parser.add_argument("--model", default=PRINTED)
    parser.add_argument("--image", default=None)
    args = parser.parse_args()

    trocr = TrOCRBaseline(model_name=args.model)
    arr = np.ones((64, 400, 3), dtype=np.uint8) * 255
    import cv2
    cv2.putText(arr, "hello world", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    res = trocr.predict(arr)
    print(f"prediction : {res['predicted_text']!r}")
    print(f"confidence : {res['confidence']}")
