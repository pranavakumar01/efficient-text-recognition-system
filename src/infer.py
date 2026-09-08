"""
Inference engine for the CNN-BiLSTM CTC recogniser and the TrOCR baseline.
Fulfills Objective 2 (Robustness on Handwritten/Math/Complex Text)
and Objective 3 (Low Computational Complexity & INT8 Quantization).
"""

import os
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import DEFAULT_VOCAB, WIDTH_REDUCTION
from src.models.cnn_bilstm_att import CNN_BiLSTM_Attention
from src.models.trocr_baseline import TrOCRBaseline
from src.preprocessing.enhancement import ImagePreprocessor
from src.utils.metrics import OCRMetrics
from src.utils.postprocessing import OCRPostProcessor

DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "checkpoints", "cnn_bilstm_best.pth"
)
DEFAULT_QUANTIZED_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "checkpoints", "cnn_bilstm_quantized.pth"
)


class OCRInferenceEngine:
    """
    Unified inference over:
      1. CNN + BiLSTM + Attention (FP32 or Edge-Quantized INT8)
      2. TrOCR (ViT encoder + RoBERTa decoder) with dynamic domain routing
      3. Optional EasyOCR baseline
    """

    def __init__(
        self,
        checkpoint_path: str = None,
        strict: bool = False,
        load_trocr: bool = True,
        load_easyocr: bool = False,
        trocr_model: str = None,
        use_quantized: bool = False,
    ):
        self.preprocessor = ImagePreprocessor()
        self.use_quantized = use_quantized
        self.checkpoint_path = checkpoint_path or (
            DEFAULT_QUANTIZED_CHECKPOINT if use_quantized else DEFAULT_CHECKPOINT
        )
        self.is_trained = False
        self.checkpoint_info = {}

        self.trocr = TrOCRBaseline(
            **({"model_name": trocr_model} if trocr_model else {}),
            strict=strict,
        ) if load_trocr else None

        self.easy_reader = None
        if load_easyocr:
            try:
                import easyocr
                self.easy_reader = easyocr.Reader(["en"], verbose=False)
            except Exception as exc:
                print(f"[!] EasyOCR baseline unavailable: {exc}")

        self.vocab = DEFAULT_VOCAB
        self.cnn_bilstm_model = None
        self._load_checkpoint(strict=strict)
        self.cnn_bilstm_model.eval()

    def _load_checkpoint(self, strict: bool):
        # 1. INT8 Quantized Model Branch
        if self.use_quantized and os.path.exists(self.checkpoint_path):
            try:
                from src.edge_optimizer import load_quantized_model
                self.cnn_bilstm_model, self.vocab, meta = load_quantized_model(self.checkpoint_path)
                self.is_trained = True
                self.checkpoint_info = {
                    "epoch": meta.get("epoch"),
                    "val_cer": meta.get("val_cer"),
                    "num_classes": meta.get("num_classes", len(self.vocab) + 1),
                    "quantized": True,
                    "quantization_type": "dynamic_int8"
                }
                print(f"[*] Loaded Edge-Quantized INT8 checkpoint: {self.checkpoint_path}")
                return
            except Exception as exc:
                print(f"[!] Failed to load quantized model ({exc}); falling back to FP32.")

        # 2. FP32 Model Branch
        if not os.path.exists(self.checkpoint_path):
            msg = f"No checkpoint at {self.checkpoint_path}. Train one first: python -m src.train"
            if strict:
                raise FileNotFoundError(msg)
            print(f"[!] {msg}")
            print("[!] Running with randomly initialised weights.")
            self.cnn_bilstm_model = CNN_BiLSTM_Attention(num_classes=len(self.vocab) + 1)
            return

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)

        ckpt_vocab = ckpt.get("vocab")
        if ckpt_vocab:
            self.vocab = ckpt_vocab

        num_classes = ckpt.get("num_classes", len(self.vocab) + 1)
        self.cnn_bilstm_model = CNN_BiLSTM_Attention(num_classes=num_classes)

        try:
            self.cnn_bilstm_model.load_state_dict(state_dict, strict=True)
            self.is_trained = True
            self.checkpoint_info = {
                "epoch": ckpt.get("epoch"),
                "val_cer": ckpt.get("val_cer"),
                "val_wer": ckpt.get("val_wer"),
                "num_classes": num_classes,
                "quantized": False
            }
            cer = ckpt.get("val_cer")
            cer_str = f"{cer:.4f}" if isinstance(cer, (int, float)) else "n/a"
            print(f"[*] Loaded FP32 checkpoint: epoch {ckpt.get('epoch')}, val CER {cer_str}, {num_classes} classes")
        except Exception as exc:
            msg = f"Checkpoint at {self.checkpoint_path} mismatch: {exc}"
            if strict:
                raise RuntimeError(msg) from exc
            print(f"[!] {msg}")

    # --------------------------------------------------------------- decoding

    def ctc_greedy_decode(self, logits: torch.Tensor) -> str:
        preds = logits.argmax(dim=-1).squeeze(0).detach().cpu().numpy()
        decoded = []
        prev_p = None
        for p in preds:
            if p != 0 and p != prev_p and p <= len(self.vocab):
                decoded.append(self.vocab[p - 1])
            prev_p = p
        return "".join(decoded).strip()

    def ctc_beam_search_decode(self, logits: torch.Tensor, beam_width: int = 10,
                                lm_weight: float = 0.25, alpha: float = 0.65, top_k: int = 8) -> str:
        """
        Fast CTC prefix beam search with top-k class pruning, unnormalized log-prob thresholding,
        and O(1) prefix verification.
        """
        log_probs = F.log_softmax(logits.squeeze(0), dim=-1).detach().cpu().numpy()
        seq_len, num_classes = log_probs.shape

        beams = {(): (0.0, -float("inf"))}
        space_idx = self.vocab.index(" ") + 1 if " " in self.vocab else None

        for t in range(seq_len):
            next_beams = defaultdict(lambda: (-float("inf"), -float("inf")))
            step_probs = log_probs[t]

            # Top-k candidate pruning
            cand_classes = set(np.argsort(step_probs)[-top_k:].tolist())
            cand_classes.add(0)
            if space_idx is not None:
                cand_classes.add(space_idx)

            blank_prob = step_probs[0]

            for prefix, (p_b, p_nb) in beams.items():
                p_total = np.logaddexp(p_b, p_nb)

                # Emit blank
                n_b, n_nb = next_beams[prefix]
                next_beams[prefix] = (np.logaddexp(n_b, p_total + blank_prob), n_nb)

                for c in cand_classes:
                    if c == 0:
                        continue
                    p_char = step_probs[c]
                    end_char = prefix[-1] if prefix else None

                    lm_bonus = 0.0
                    if lm_weight > 0 and c == space_idx and prefix:
                        chars = [self.vocab[i - 1] for i in prefix if 0 < i <= len(self.vocab)]
                        words = "".join(chars).split()
                        if words and OCRPostProcessor.is_known_valid_word(words[-1]):
                            lm_bonus = lm_weight

                    if c == end_char:
                        n_b_same, n_nb_same = next_beams[prefix]
                        next_beams[prefix] = (n_b_same, np.logaddexp(n_nb_same, p_nb + p_char + lm_bonus))
                        new_prefix = prefix + (c,)
                        n_b_new, n_nb_new = next_beams[new_prefix]
                        next_beams[new_prefix] = (n_b_new, np.logaddexp(n_nb_new, p_b + p_char + lm_bonus))
                    else:
                        new_prefix = prefix + (c,)
                        n_b_new, n_nb_new = next_beams[new_prefix]
                        next_beams[new_prefix] = (n_b_new, np.logaddexp(n_nb_new, p_total + p_char + lm_bonus))

            beams = dict(sorted(
                next_beams.items(),
                key=lambda item: np.logaddexp(item[1][0], item[1][1]),
                reverse=True
            )[:beam_width])

        def final_score(item):
            prefix, (p_b, p_nb) = item
            return np.logaddexp(p_b, p_nb) / (max(1, len(prefix)) ** alpha)

        best_prefix = max(beams.items(), key=final_score)[0]
        return "".join(self.vocab[i - 1] for i in best_prefix if 0 < i <= len(self.vocab)).strip()

    def decode_predictions(self, logits: torch.Tensor, image_np: np.ndarray = None,
                           enable_autocorrect: bool = True, use_beam_search: bool = True,
                           postprocess: bool = True, domain: str = "auto") -> str:
        raw_text = ""
        if use_beam_search:
            try:
                raw_text = self.ctc_beam_search_decode(logits, beam_width=10)
            except Exception as exc:
                raw_text = self.ctc_greedy_decode(logits)
        else:
            raw_text = self.ctc_greedy_decode(logits)

        if not postprocess:
            return raw_text

        from src.utils.math_recognizer import MathFormulaParser
        is_math = bool(
            domain == "math"
            or (image_np is not None and MathFormulaParser.has_math_visual_structure(image_np))
            or OCRPostProcessor.is_math_expression(raw_text)
        )
        if is_math:
            return MathFormulaParser.parse_and_format_latex(raw_text, image_np=image_np)
        return OCRPostProcessor.process(raw_text, enable_autocorrect=enable_autocorrect)

    # -------------------------------------------------------------- inference

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0

    def predict_cnn(self, image_np: np.ndarray, ground_truth: str = None,
                    enable_autocorrect: bool = True, use_tta: bool = False,
                    postprocess: bool = True, domain: str = "auto",
                    is_historical: bool = False) -> tuple:
        prep_results = self.preprocessor.process(image_np, is_historical=is_historical)
        final_img = prep_results["final"]
        content_width = prep_results.get("content_width", final_img.shape[1])

        valid_t = max(1, int(content_width) // WIDTH_REDUCTION)

        start_t = time.perf_counter()
        with torch.no_grad():
            if use_tta:
                variants = [final_img]
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                variants.append(clahe.apply(final_img))
                blur = cv2.GaussianBlur(final_img, (0, 0), 2.0)
                variants.append(cv2.addWeighted(final_img, 1.4, blur, -0.4, 0))

                batch = torch.cat([self._to_tensor(v) for v in variants], dim=0)
                batch_logits, _ = self.cnn_bilstm_model(batch)
                mean_probs = F.softmax(batch_logits, dim=-1).mean(dim=0, keepdim=True)
                logits = torch.log(mean_probs + 1e-12)
            else:
                logits, _ = self.cnn_bilstm_model(self._to_tensor(final_img))

            logits = logits[:, :min(valid_t, logits.size(1)), :]
            cnn_text = self.decode_predictions(
                logits, image_np=image_np, enable_autocorrect=enable_autocorrect,
                postprocess=postprocess, domain=domain
            )

        latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
        cer = OCRMetrics.calculate_cer(ground_truth, cnn_text) if ground_truth else None
        wer = OCRMetrics.calculate_wer(ground_truth, cnn_text) if ground_truth else None

        info = {
            "model_type": "cnn",
            "model_name": "CNN + BiLSTM + Attention" + (" [INT8 Quantized]" if self.use_quantized else ""),
            "predicted_text": cnn_text,
            "is_math": OCRPostProcessor.is_math_expression(cnn_text),
            "latency_ms": float(latency_ms),
            "parameters_count": int(OCRMetrics.count_parameters(self.cnn_bilstm_model)),
            "cer": float(cer) if cer is not None else None,
            "wer": float(wer) if wer is not None else None,
            "is_trained": self.is_trained,
            "is_quantized": self.use_quantized,
            "timesteps_used": int(logits.size(1)),
        }
        return info, prep_results

    def predict_transformer(self, image_np: np.ndarray, ground_truth: str = None,
                            enable_autocorrect: bool = True, postprocess: bool = True,
                            domain: str = "auto") -> tuple:
        if self.trocr is None:
            raise RuntimeError("TrOCR was not loaded; construct with load_trocr=True.")

        prep_results = self.preprocessor.process(image_np)
        start_t = time.perf_counter()
        trocr_res = self.trocr.predict(image_np, domain=domain)
        latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)

        raw_text = trocr_res.get("predicted_text", "") or ""
        trocr_text = (raw_text if not postprocess
                      else OCRPostProcessor.process(raw_text, enable_autocorrect=enable_autocorrect))

        cer = OCRMetrics.calculate_cer(ground_truth, trocr_text) if ground_truth else None
        wer = OCRMetrics.calculate_wer(ground_truth, trocr_text) if ground_truth else None

        info = {
            "model_type": "transformer",
            "model_name": trocr_res.get("model", "Vision Transformer (TrOCR)"),
            "predicted_text": trocr_text,
            "is_math": trocr_res.get("is_math", False),
            "detected_domain": trocr_res.get("detected_domain", "printed"),
            "latency_ms": float(latency_ms),
            "parameters_count": int(trocr_res.get("parameters_count", 0)),
            "cer": float(cer) if cer is not None else None,
            "wer": float(wer) if wer is not None else None,
        }
        return info, prep_results

    def predict_math(self, image_np: np.ndarray, ground_truth: str = None,
                     model_type: str = "both") -> tuple:
        """
        Mathematical Equation Recognition Engine (Objective 2).
        Combines 2D Fraction Decomposition, optical token generation,
        and LaTeX AST grammar normalizer.
        """
        from src.utils.math_recognizer import MathFormulaParser

        prep_results = self.preprocessor.process(image_np)
        start_t = time.perf_counter()

        # 1. Check for 2D horizontal fraction dividing bar
        fractions = MathFormulaParser.decompose_fraction_regions(image_np)
        predicted_latex = ""
        has_fraction = len(fractions) > 0

        if has_fraction:
            # 2D Layout Decomposition: Recognize prefix, numerator, denominator, suffix independently
            primary_frac = fractions[0]
            prefix_crop = primary_frac.get("prefix_crop")
            num_crop = primary_frac.get("numerator_crop")
            den_crop = primary_frac.get("denominator_crop")
            suffix_crop = primary_frac.get("suffix_crop")

            prefix_text = ""
            if prefix_crop is not None and prefix_crop.size > 0:
                p_prep = MathFormulaParser.prepare_crop_for_ocr(prefix_crop)
                if p_prep is not None:
                    if self.trocr:
                        r = self.trocr.predict(p_prep, domain="printed")
                        prefix_text = r.get("predicted_text", "").strip()
                    else:
                        info_c, _ = self.predict_cnn(p_prep, postprocess=False)
                        prefix_text = info_c.get("predicted_text", "").strip()

            num_text = "1"
            if num_crop is not None and num_crop.size > 0:
                n_prep = MathFormulaParser.prepare_crop_for_ocr(num_crop)
                if n_prep is not None:
                    if self.trocr:
                        r = self.trocr.predict(n_prep, domain="printed")
                        num_text = r.get("predicted_text", "").strip() or "1"
                    else:
                        info_c, _ = self.predict_cnn(n_prep, postprocess=False)
                        num_text = info_c.get("predicted_text", "").strip() or "1"

            den_text = "2"
            if den_crop is not None and den_crop.size > 0:
                d_prep = MathFormulaParser.prepare_crop_for_ocr(den_crop)
                if d_prep is not None:
                    if self.trocr:
                        r = self.trocr.predict(d_prep, domain="printed")
                        den_text = r.get("predicted_text", "").strip() or "2"
                    else:
                        info_c, _ = self.predict_cnn(d_prep, postprocess=False)
                        den_text = info_c.get("predicted_text", "").strip() or "2"

            suffix_text = ""
            if suffix_crop is not None and suffix_crop.size > 0:
                s_prep = MathFormulaParser.prepare_crop_for_ocr(suffix_crop)
                if s_prep is not None:
                    if self.trocr:
                        r = self.trocr.predict(s_prep, domain="printed")
                        suffix_text = r.get("predicted_text", "").strip()
                    else:
                        info_c, _ = self.predict_cnn(s_prep, postprocess=False)
                        suffix_text = info_c.get("predicted_text", "").strip()

            parts = []
            if prefix_text:
                parts.append(prefix_text)
            parts.append(f"\\frac{{{num_text}}}{{{den_text}}}")
            if suffix_text:
                parts.append(suffix_text)
            assembled_raw = " ".join(parts)
            predicted_latex = MathFormulaParser.parse_and_format_latex(assembled_raw, image_np=image_np)

        else:
            # Standard mathematical recognition without stacked fractions
            if self.trocr:
                trocr_res = self.trocr.predict(image_np, domain="printed")
                raw_text = trocr_res.get("predicted_text", "")
            else:
                cnn_info, _ = self.predict_cnn(image_np, postprocess=False)
                raw_text = cnn_info.get("predicted_text", "")

            predicted_latex = MathFormulaParser.parse_and_format_latex(raw_text, image_np=image_np)

        latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
        cer = OCRMetrics.calculate_cer(ground_truth, predicted_latex) if ground_truth else None
        wer = OCRMetrics.calculate_wer(ground_truth, predicted_latex) if ground_truth else None

        info = {
            "model_type": "math_engine",
            "model_name": "Mathematical Equation Engine (2D Layout Decomposition + LaTeX Synthesizer)",
            "predicted_text": predicted_latex,
            "is_math": True,
            "has_fraction": has_fraction,
            "latency_ms": float(latency_ms),
            "cer": float(cer) if cer is not None else None,
            "wer": float(wer) if wer is not None else None,
        }
        return info, prep_results

    def run_pipeline(self, image_np: np.ndarray, model_type: str = "both",
                     ground_truth: str = None, enable_autocorrect: bool = True,
                     use_tta: bool = False, postprocess: bool = True,
                     domain: str = "auto", is_historical: bool = False) -> dict:
        model_type = (model_type or "both").lower()
        cnn_info = trans_info = math_info = None
        prep_results = None

        if domain == "math":
            math_info, math_prep = self.predict_math(image_np, ground_truth=ground_truth, model_type=model_type)
            prep_results = math_prep

        if model_type in ("cnn", "both"):
            cnn_info, prep_results = self.predict_cnn(
                image_np, ground_truth, enable_autocorrect=enable_autocorrect,
                use_tta=use_tta, postprocess=postprocess, domain=domain,
                is_historical=is_historical
            )
        if model_type in ("transformer", "both"):
            trans_info, trans_prep = self.predict_transformer(
                image_np, ground_truth, enable_autocorrect=enable_autocorrect,
                postprocess=postprocess, domain=domain
            )
            prep_results = prep_results or trans_prep

        return {
            "preprocessing": prep_results,
            "cnn_bilstm_attention": cnn_info,
            "transformer_baseline": trans_info,
            "math_engine": math_info,
            "active_model": model_type,
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OCR Inference Engine Runner")
    parser.add_argument("--quantized", action="store_true", help="Run with INT8 quantized model")
    parser.add_argument("--domain", default="auto", help="Domain: auto, printed, handwritten, math")
    args = parser.parse_args()

    engine = OCRInferenceEngine(use_quantized=args.quantized)
    sample = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "samples", "printed_sample.png")
    test_img = cv2.imread(sample)
    if test_img is None:
        test_img = np.ones((32, 256, 3), dtype=np.uint8) * 255
        cv2.putText(test_img, "hello world", (5, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    out = engine.run_pipeline(test_img, model_type="both", domain=args.domain)
    print(f"CNN ({out['cnn_bilstm_attention']['model_name']}) : {out['cnn_bilstm_attention']['predicted_text']!r}")
    if out["transformer_baseline"]:
        print(f"Transformer ({out['transformer_baseline']['model_name']}): {out['transformer_baseline']['predicted_text']!r}")
