"""
Multi-Domain End-to-End Training Pipeline for Both CNN and Transformer OCR Models.

Trains the entire dataset with respect to:
  1. Historical Documents (aged/degraded paper, manuscript restoration, archival texts)
  2. Mathematical Equations (Google MathWriting, LaTeX notation, fractions, radicals)
  3. Handwritten Texts (Kaggle French handwriting, cursive scripts, stroke variations)
  4. Printed & Synthetic Texts (high-accuracy clean typography)

Models Supported:
  - CNN Architecture: CNN + BiLSTM + Temporal Attention + CTC Loss
    * Features: SimCLR SSL backbone weight transfer, key-padding attention masking,
      true-width timestep alignment, Cosine Annealing with warmup, and dynamic INT8 edge quantization.
  - Transformer Architecture: VisionEncoderDecoder (TrOCR)
    * Features: Sequence-to-sequence autoregressive cross-attention, arbitrary 2D math
      layout learning, subword tokenization, AdamW with OneCycleLR/Cosine scheduling.

Usage:
  # Train both models on the entire dataset with best accuracy:
  python train_all.py

  # Train only the CNN model:
  python train_all.py --model cnn --cnn-epochs 40

  # Train only the Transformer model:
  python train_all.py --model transformer --transformer-epochs 5

  # Train specifically on historical documents:
  python train_all.py --domain historical

  # Train specifically on mathematical equations:
  python train_all.py --domain math

  # Quick smoke-test (verifies entire pipeline across all domains):
  python train_all.py --max-samples 16 --cnn-epochs 1 --transformer-epochs 1
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Ensure project root is in python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import (
    DEFAULT_MANIFEST,
    DEFAULT_VOCAB,
    WIDTH_REDUCTION,
    Tokenizer,
    build_vocab,
)
from src.edge_optimizer import quantize_model, export_quantized_checkpoint
from src.models.cnn_bilstm_att import CNN_BiLSTM_Attention
from src.preprocessing.enhancement import ImagePreprocessor
from src.utils.metrics import OCRMetrics

# Domain Keywords & Classification Helpers
HISTORICAL_KEYWORDS = {
    "historical", "archival", "archive", "archives", "ancient", "heritage",
    "century", "manuscript", "bleed", "degraded", "restoration", "ink",
    "irish", "gaelic", "parchment", "faded", "binarization", "deskewing",
    "papyrus", "medieval", "calligraphy", "scroll",
}
MATH_CHARS = set(r"\^_{}[]()=+-/*<>|%")


def classify_domain(row: dict) -> str:
    """
    Classifies a manifest row into 'historical', 'mathematical', 'handwritten', or 'printed'.
    Correctly orders historical precedence over generic folder matches to prevent misclassification.
    """
    ds = row.get("dataset", "").lower()
    is_math = str(row.get("is_math", "0")).strip() == "1"
    path = str(row.get("image_path", "")).replace("\\", "/").lower()
    lbl = str(row.get("label", "")).lower()

    if ds == "mathwriting" or is_math:
        return "mathematical"
    if "hist" in path or any(re.search(r"\b" + re.escape(w) + r"\b", lbl) for w in HISTORICAL_KEYWORDS):
        return "historical"
    if ds == "kaggle_dataset" or "handwritten" in path or "handwritten" in lbl:
        return "handwritten"
    return "printed"


def load_trocr_processor_and_tokenizer(model_name_or_path: str):
    """
    Robustly loads the appropriate HuggingFace TrOCR processor and tokenizer.
    Correctly selects XLMRobertaTokenizer for 'microsoft/trocr-small-printed' (SentencePiece)
    and RobertaTokenizer for 'microsoft/trocr-base-*' models to prevent token corruption.
    """
    from transformers import (
        AutoTokenizer,
        RobertaTokenizer,
        TrOCRProcessor,
        ViTImageProcessor,
        XLMRobertaTokenizer,
    )

    tok_class = None
    cfg_file = (
        os.path.join(model_name_or_path, "tokenizer_config.json")
        if os.path.isdir(model_name_or_path)
        else None
    )

    if cfg_file and os.path.exists(cfg_file):
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                tok_class = json.load(f).get("tokenizer_class")
        except Exception:
            pass
    elif not cfg_file:
        try:
            from huggingface_hub import hf_hub_download
            hf_cfg = hf_hub_download(model_name_or_path, "tokenizer_config.json")
            with open(hf_cfg, "r", encoding="utf-8") as f:
                tok_class = json.load(f).get("tokenizer_class")
        except Exception:
            pass

    # Instantiate the correct tokenizer backend
    if tok_class == "XLMRobertaTokenizer" or "small-printed" in model_name_or_path.lower():
        try:
            tokenizer = XLMRobertaTokenizer.from_pretrained(model_name_or_path)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    elif tok_class == "RobertaTokenizer" or "roberta" in (tok_class or "").lower():
        try:
            tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        except Exception:
            try:
                tokenizer = XLMRobertaTokenizer.from_pretrained(model_name_or_path)
            except Exception:
                tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)

    image_processor = ViTImageProcessor.from_pretrained(model_name_or_path)
    return TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)


def setup_trocr_model(model, processor):
    """
    Properly aligns TrOCR model config, decoder config, and generation config.
    Ensures decoder_start_token_id is set to eos_token_id (token 2) as configured by Microsoft.
    """
    tok = processor.tokenizer
    start_tok = getattr(model.config.decoder, "decoder_start_token_id", None)
    if start_tok is None:
        start_tok = tok.eos_token_id if tok.eos_token_id is not None else 2

    pad_tok = tok.pad_token_id if tok.pad_token_id is not None else 1
    eos_tok = tok.eos_token_id if tok.eos_token_id is not None else 2

    model.config.decoder_start_token_id = start_tok
    model.config.pad_token_id = pad_tok
    model.config.eos_token_id = eos_tok
    model.config.vocab_size = getattr(model.config.decoder, "vocab_size", len(tok))

    if hasattr(model.config, "decoder") and model.config.decoder is not None:
        model.config.decoder.decoder_start_token_id = start_tok
        model.config.decoder.pad_token_id = pad_tok
        model.config.decoder.eos_token_id = eos_tok

    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.decoder_start_token_id = start_tok
        model.generation_config.pad_token_id = pad_tok
        model.generation_config.eos_token_id = eos_tok

    return model


# =============================================================================
# 1. DOMAIN-AWARE DATASET FOR CNN (CNN-BiLSTM-Attention with CTC)
# =============================================================================

class DomainOCRDataset(Dataset):
    """
    CNN-BiLSTM-Attention Dataset supporting multi-domain text recognition.
    Integrates domain-stratified sampling, domain-specific enhancement (historical
    illumination correction, handwriting stroke jitter), and true content width calculation.
    """

    def __init__(
        self,
        manifest: str = DEFAULT_MANIFEST,
        split: str = "train",
        data_root: str = None,
        domain_filter: str = "all",
        include_math: bool = True,
        exclude_infeasible: bool = True,
        augment: bool = False,
        target_h: int = 32,
        max_width: int = 640,
        tokenizer: Optional[Tokenizer] = None,
        max_samples: Optional[int] = None,
    ):
        self.manifest = manifest
        self.split = split
        self.data_root = data_root or os.path.dirname(os.path.abspath(manifest))
        self.domain_filter = domain_filter.lower()
        self.include_math = include_math
        self.exclude_infeasible = exclude_infeasible
        self.augment = augment
        self.target_h = target_h
        self.max_width = max_width
        self.tokenizer = tokenizer or Tokenizer()
        self.preprocessor = ImagePreprocessor(target_height=target_h, default_max_width=max_width)

        self.samples = []
        self._load_manifest(max_samples)

    def _load_manifest(self, max_samples: Optional[int]):
        if not os.path.exists(self.manifest):
            raise FileNotFoundError(
                f"[!] Manifest not found at '{self.manifest}'. Run 'python -m src.make_splits' first."
            )

        with open(self.manifest, "r", encoding="utf-8", newline="") as f:
            all_rows = list(csv.DictReader(f))

        candidates = []
        domain_buckets = defaultdict(list)

        for row in all_rows:
            if self.split and row.get("split", "").lower() != self.split.lower():
                continue

            domain = classify_domain(row)
            if self.domain_filter not in ("all", "both"):
                if self.domain_filter == "math" and domain != "mathematical":
                    continue
                elif self.domain_filter != "math" and domain != self.domain_filter:
                    continue

            is_math = domain == "mathematical"
            feasible = str(row.get("ctc_feasible", "1")) == "1"

            if is_math and not self.include_math:
                continue
            if self.exclude_infeasible and not feasible:
                continue

            rel_path = (row.get("image_path") or "").replace("\\", "/")
            abs_path = os.path.join(self.data_root, os.path.normpath(rel_path))
            if not os.path.exists(abs_path):
                continue

            sample_entry = {
                "path": abs_path,
                "label": row["label"],
                "domain": domain,
                "ctc_feasible": feasible,
                "timesteps": int(row.get("timesteps") or 0),
            }
            candidates.append(sample_entry)
            domain_buckets[domain].append(sample_entry)

        if max_samples is not None and max_samples < len(candidates):
            # Domain-stratified sampling: ensure all 4 domains are balanced
            if self.domain_filter in ("all", "both") and len(domain_buckets) > 1:
                target_per_dom = max(1, max_samples // len(domain_buckets))
                stratified = []
                for dom in ["historical", "mathematical", "handwritten", "printed"]:
                    stratified.extend(domain_buckets[dom][:target_per_dom])
                # Fill remaining slots if needed
                if len(stratified) < max_samples:
                    for dom in ["historical", "mathematical", "handwritten", "printed"]:
                        extra = domain_buckets[dom][target_per_dom:]
                        needed = max_samples - len(stratified)
                        stratified.extend(extra[:needed])
                        if len(stratified) >= max_samples:
                            break
                self.samples = stratified[:max_samples]
            else:
                self.samples = candidates[:max_samples]
        else:
            self.samples = candidates

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def labels(self) -> List[str]:
        return [s["label"] for s in self.samples]

    def augment_image(self, img: np.ndarray, domain: str) -> np.ndarray:
        """Domain-adaptive geometric & photometric jitter."""
        bg_val = int(np.median(img)) if img.size > 0 else 255
        h, w = img.shape[:2]

        if domain == "handwritten":
            if np.random.rand() > 0.4:
                angle = np.random.uniform(-4.0, 4.0)
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=bg_val)
            if np.random.rand() > 0.6:
                img = cv2.GaussianBlur(img, (3, 3), 0.6)
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.85, 1.15)
                beta = np.random.uniform(-15, 15)
                img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
            return img

        if domain == "historical":
            if np.random.rand() > 0.4:
                noise = np.random.normal(0, np.random.uniform(3, 8), img.shape).astype(np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-2.0, 2.0)
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=bg_val)
            if np.random.rand() > 0.6:
                img = cv2.GaussianBlur(img, (3, 3), 0.5)
            return img

        # Standard printed & mathematical text jitter
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-1.5, 1.5)
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=bg_val)
        if np.random.rand() > 0.7:
            img = cv2.GaussianBlur(img, (3, 3), 0.4)
        return img

    def preprocess(self, img_path: str, domain: str) -> Tuple[torch.Tensor, int]:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.ones((self.target_h, 128), dtype=np.uint8) * 255

        if self.augment:
            img = self.augment_image(img, domain)

        is_hist = (domain == "historical")
        final_img, content_width = self.preprocessor.prepare(
            img, max_width=self.max_width, is_historical=is_hist
        )
        tensor = torch.from_numpy(final_img).float().unsqueeze(0) / 255.0
        return tensor, int(content_width)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        tensor, content_width = self.preprocess(s["path"], s["domain"])
        encoded = self.tokenizer.encode(s["label"])
        return (
            tensor,
            torch.tensor(encoded, dtype=torch.long),
            s["label"],
            content_width,
            s["domain"],
        )


def domain_ocr_collate_fn(batch):
    """Batches images with dynamic padding to multiples of 4 and per-sample CTC lengths."""
    images, target_tensors, label_texts, content_widths, domains = zip(*batch)

    h = images[0].shape[1]
    max_w = max(img.shape[2] for img in images)
    max_w = int(np.ceil(max_w / WIDTH_REDUCTION) * WIDTH_REDUCTION)

    padded = []
    for img in images:
        pad_w = max_w - img.shape[2]
        if pad_w > 0:
            pad = torch.ones((1, h, pad_w), dtype=torch.float32)
            img = torch.cat([img, pad], dim=2)
        padded.append(img)

    images_batch = torch.stack(padded, dim=0)
    target_lengths = torch.tensor([len(t) for t in target_tensors], dtype=torch.long)
    targets = torch.cat(target_tensors, dim=0) if len(target_tensors) > 0 else torch.tensor([], dtype=torch.long)

    timesteps_available = max_w // WIDTH_REDUCTION
    input_lengths = torch.tensor(
        [max(1, min(timesteps_available, w // WIDTH_REDUCTION)) for w in content_widths],
        dtype=torch.long,
    )
    return images_batch, targets, target_lengths, label_texts, input_lengths, domains


# =============================================================================
# 2. DOMAIN-AWARE DATASET FOR TRANSFORMER (TrOCR / VisionEncoderDecoder)
# =============================================================================

class DomainTrOCRDataset(Dataset):
    """
    Vision Transformer (TrOCR) Dataset supporting multi-domain text recognition.
    Directly provides RGB crops with subword tokens and domain labeling.
    Supports historical document illumination restoration and training augmentation.
    """

    def __init__(
        self,
        processor,
        manifest: str = DEFAULT_MANIFEST,
        split: str = "train",
        data_root: str = None,
        domain_filter: str = "all",
        include_math: bool = True,
        max_target_length: int = 128,
        augment: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.processor = processor
        self.manifest = manifest
        self.split = split
        self.data_root = data_root or os.path.dirname(os.path.abspath(manifest))
        self.domain_filter = domain_filter.lower()
        self.include_math = include_math
        self.max_target_length = max_target_length
        self.augment = augment
        self.preprocessor = ImagePreprocessor()

        self.samples = []
        self._load_manifest(max_samples)

    def _load_manifest(self, max_samples: Optional[int]):
        if not os.path.exists(self.manifest):
            raise FileNotFoundError(f"[!] Manifest not found at '{self.manifest}'")

        with open(self.manifest, "r", encoding="utf-8", newline="") as f:
            all_rows = list(csv.DictReader(f))

        candidates = []
        domain_buckets = defaultdict(list)

        for row in all_rows:
            if self.split and row.get("split", "").lower() != self.split.lower():
                continue

            domain = classify_domain(row)
            if self.domain_filter not in ("all", "both"):
                if self.domain_filter == "math" and domain != "mathematical":
                    continue
                elif self.domain_filter != "math" and domain != self.domain_filter:
                    continue

            is_math = domain == "mathematical"
            if is_math and not self.include_math:
                continue

            rel_path = (row.get("image_path") or "").replace("\\", "/")
            abs_path = os.path.join(self.data_root, os.path.normpath(rel_path))
            if not os.path.exists(abs_path):
                continue

            sample_entry = {
                "path": abs_path,
                "label": row["label"],
                "domain": domain,
            }
            candidates.append(sample_entry)
            domain_buckets[domain].append(sample_entry)

        if max_samples is not None and max_samples < len(candidates):
            # Domain-stratified sampling: ensure all domains are represented
            if self.domain_filter in ("all", "both") and len(domain_buckets) > 1:
                target_per_dom = max(1, max_samples // len(domain_buckets))
                stratified = []
                for dom in ["historical", "mathematical", "handwritten", "printed"]:
                    stratified.extend(domain_buckets[dom][:target_per_dom])
                if len(stratified) < max_samples:
                    for dom in ["historical", "mathematical", "handwritten", "printed"]:
                        extra = domain_buckets[dom][target_per_dom:]
                        needed = max_samples - len(stratified)
                        stratified.extend(extra[:needed])
                        if len(stratified) >= max_samples:
                            break
                self.samples = stratified[:max_samples]
            else:
                self.samples = candidates[:max_samples]
        else:
            self.samples = candidates

    def __len__(self) -> int:
        return len(self.samples)

    def _prepare_image(self, path: str, domain: str) -> Image.Image:
        img = cv2.imread(path)
        if img is None:
            return Image.new("RGB", (384, 32), (255, 255, 255))

        # Domain-aware preprocessing for historical aged paper
        if domain == "historical":
            img = self.preprocessor.enhance_historical(img)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.augment:
            h, w = img.shape[:2]
            bg_c = (255, 255, 255)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-2.5, 2.5)
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=bg_c)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return Image.fromarray(img)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = self._prepare_image(s["path"], s["domain"])
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values[0]

        ids = self.processor.tokenizer(
            s["label"],
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
        ).input_ids

        pad_id = self.processor.tokenizer.pad_token_id
        labels = [tok if tok != pad_id else -100 for tok in ids]

        return (
            pixel_values,
            torch.tensor(labels, dtype=torch.long),
            s["label"],
            s["domain"],
        )


def trocr_collate_fn(batch):
    pixel_values = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    texts = [b[2] for b in batch]
    domains = [b[3] for b in batch]
    return pixel_values, labels, texts, domains


# =============================================================================
# 3. DOMAIN-STRATIFIED EVALUATION HARNESSES
# =============================================================================

def evaluate_cnn_split(model, loader, criterion, tokenizer, device) -> dict:
    """Evaluates CNN model with domain-stratified CER, WER, and Exact Match % breakdowns."""
    model.eval()
    total_loss, n_loss = 0.0, 0
    all_cers, all_wers, exact_matches = [], [], []
    domain_cers = defaultdict(list)
    domain_wers = defaultdict(list)
    domain_em = defaultdict(list)
    examples = []

    with torch.no_grad():
        for images, targets, target_lengths, label_texts, input_lengths, domains in loader:
            images = images.to(device)
            clamped = input_lengths.clamp(max=images.size(3) // WIDTH_REDUCTION).to(device)
            logits, _ = model(images, input_lengths=clamped)

            seq_len = logits.size(1)
            clamped = clamped.clamp(max=seq_len)

            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
            loss = criterion(log_probs, targets.to(device), clamped, target_lengths.to(device))
            if torch.isfinite(loss):
                total_loss += loss.item() * images.size(0)
                n_loss += images.size(0)

            # Greedy CTC decode
            preds = logits.argmax(dim=-1).cpu()
            for i in range(preds.size(0)):
                valid_t = int(clamped[i])
                pred_text = tokenizer.decode(preds[i, :valid_t].tolist()).strip()
                truth = label_texts[i].strip()
                dom = domains[i]

                cer = OCRMetrics.calculate_cer(truth, pred_text)
                wer = OCRMetrics.calculate_wer(truth, pred_text)
                em = 1.0 if pred_text == truth else 0.0

                all_cers.append(cer)
                all_wers.append(wer)
                exact_matches.append(em)

                domain_cers[dom].append(cer)
                domain_wers[dom].append(wer)
                domain_em[dom].append(em)

                if len(examples) < 8:
                    examples.append((dom, truth, pred_text))

    n = max(1, len(all_cers))
    mean_loss = total_loss / max(1, n_loss)
    mean_cer = sum(all_cers) / n
    mean_wer = sum(all_wers) / n
    mean_em = sum(exact_matches) / n

    per_domain = {}
    for dom in ("historical", "mathematical", "handwritten", "printed"):
        c_list = domain_cers.get(dom, [])
        w_list = domain_wers.get(dom, [])
        e_list = domain_em.get(dom, [])
        if c_list:
            per_domain[dom] = {
                "n": len(c_list),
                "cer": sum(c_list) / len(c_list),
                "wer": sum(w_list) / len(w_list),
                "em": sum(e_list) / len(e_list),
            }

    return {
        "loss": mean_loss,
        "cer": mean_cer,
        "wer": mean_wer,
        "exact_match": mean_em,
        "total_samples": len(all_cers),
        "per_domain": per_domain,
        "examples": examples,
    }


@torch.no_grad()
def evaluate_transformer_split(
    model, processor, loader, device, max_new_tokens: int = 96, num_beams: int = 1
) -> dict:
    """Evaluates Vision Transformer with domain-stratified CER, WER, and Exact Match %."""
    model.eval()
    all_cers, all_wers, exact_matches = [], [], []
    domain_cers = defaultdict(list)
    domain_wers = defaultdict(list)
    domain_em = defaultdict(list)
    examples = []

    for pixel_values, _, texts, domains in loader:
        generated = model.generate(
            pixel_values.to(device),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        preds = processor.batch_decode(generated, skip_special_tokens=True)

        for pred, truth, dom in zip(preds, texts, domains):
            pred = pred.strip()
            truth = truth.strip()
            cer = OCRMetrics.calculate_cer(truth, pred)
            wer = OCRMetrics.calculate_wer(truth, pred)
            em = 1.0 if pred == truth else 0.0

            all_cers.append(cer)
            all_wers.append(wer)
            exact_matches.append(em)

            domain_cers[dom].append(cer)
            domain_wers[dom].append(wer)
            domain_em[dom].append(em)

            if len(examples) < 8:
                examples.append((dom, truth, pred))

    n = max(1, len(all_cers))
    mean_cer = sum(all_cers) / n
    mean_wer = sum(all_wers) / n
    mean_em = sum(exact_matches) / n

    per_domain = {}
    for dom in ("historical", "mathematical", "handwritten", "printed"):
        c_list = domain_cers.get(dom, [])
        w_list = domain_wers.get(dom, [])
        em_list = domain_em.get(dom, [])
        if c_list:
            per_domain[dom] = {
                "n": len(c_list),
                "cer": sum(c_list) / len(c_list),
                "wer": sum(w_list) / len(w_list),
                "em": sum(em_list) / len(em_list),
            }

    return {
        "cer": mean_cer,
        "wer": mean_wer,
        "exact_match": mean_em,
        "total_samples": len(all_cers),
        "per_domain": per_domain,
        "examples": examples,
    }


# =============================================================================
# 4. CNN-BiLSTM-ATTENTION TRAINING WORKFLOW
# =============================================================================

def train_cnn_model(
    manifest: str,
    data_root: str,
    save_dir: str,
    domain: str = "all",
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 1e-3,
    patience: int = 10,
    use_ssl: bool = True,
    include_math: bool = True,
    warmup_steps: int = 200,
    max_samples: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[str, dict]:
    """Supervised CTC training for CNN-BiLSTM-Attention across all specified domains."""
    print("\n" + "=" * 78)
    print(" [STAGE 1/2] TRAINING PRIMARY ARCHITECTURE: CNN + BiLSTM + ATTENTION")
    print("=" * 78)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Dataset & Comprehensive Vocabulary preparation across entire dataset
    with open(manifest, "r", encoding="utf-8", newline="") as f:
        manifest_rows = list(csv.DictReader(f))
    all_labels = [r["label"] for r in manifest_rows if r.get("label")]
    vocab = build_vocab(all_labels, include_common_punctuation=True)
    if include_math:
        math_extra = "^_{}[]()=+-/*<>|%\\$#@&~`'\""
        vocab = "".join(sorted(set(vocab).union(set(math_extra))))

    tokenizer = Tokenizer(vocab=vocab)
    num_classes = len(tokenizer)

    train_ds = DomainOCRDataset(
        manifest=manifest, split="train", data_root=data_root, domain_filter=domain,
        include_math=include_math, exclude_infeasible=True, augment=True,
        tokenizer=tokenizer, max_samples=max_samples,
    )
    val_ds = DomainOCRDataset(
        manifest=manifest, split="valid", data_root=data_root, domain_filter=domain,
        include_math=include_math, exclude_infeasible=True, augment=False,
        tokenizer=tokenizer,
        max_samples=max_samples if max_samples is None else max(8, max_samples // 2),
    )

    if len(train_ds) == 0:
        raise ValueError(f"[!] No valid training samples found for domain='{domain}'")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=domain_ocr_collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=domain_ocr_collate_fn, num_workers=0)

    print(f"  Domain scope        : {domain.upper()} (historical, math, handwritten, printed)")
    print(f"  Train / Valid split : {len(train_ds)} / {len(val_ds)} samples")
    print(f"  Vocabulary size     : {num_classes} classes (including blank) from {len(vocab)} chars")
    print(f"  Epochs / Batch / LR : {epochs} / {batch_size} / {lr}")
    print(f"  Device              : {device}")

    # 2. Model initialization & SSL pre-trained weights
    model = CNN_BiLSTM_Attention(num_classes=num_classes).to(device)

    ssl_loaded = False
    if use_ssl:
        ssl_path = os.path.join(save_dir, "ssl_backbone_best.pth")
        if os.path.exists(ssl_path):
            ssl_loaded = model.load_ssl_weights(ssl_path)

    criterion = nn.CTCLoss(blank=tokenizer.blank_idx, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)

    best_cer = float("inf")
    best_epoch = 0
    epochs_since_best = 0
    best_ckpt_name = "cnn_bilstm_best.pth" if max_samples is None else "cnn_bilstm_smoketest.pth"
    best_path = os.path.join(save_dir, best_ckpt_name)

    history = []
    global_step = 0

    print("\n[*] Commencing CNN CTC Optimization...")
    for epoch in range(1, epochs + 1):
        start_t = time.time()
        model.train()
        running_loss, seen = 0.0, 0

        for batch_idx, (images, targets, target_lengths, _, input_lengths, _) in enumerate(train_loader, 1):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            global_step += 1
            if warmup_steps and global_step <= warmup_steps:
                for pg in optimizer.param_groups:
                    pg["lr"] = lr * global_step / warmup_steps

            optimizer.zero_grad()
            clamped = input_lengths.clamp(max=images.size(3) // WIDTH_REDUCTION).to(device)
            logits, _ = model(images, input_lengths=clamped)
            seq_len = logits.size(1)
            clamped = clamped.clamp(max=seq_len)

            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
            loss = criterion(log_probs, targets, clamped, target_lengths)

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                seen += images.size(0)

        train_loss = running_loss / max(1, seen)
        val_metrics = evaluate_cnn_split(model, val_loader, criterion, tokenizer, device)
        scheduler.step()
        elapsed = round(time.time() - start_t, 1)

        val_cer = val_metrics["cer"]
        val_wer = val_metrics["wer"]
        val_em = val_metrics["exact_match"]
        val_loss = val_metrics["loss"]

        flag = ""
        if val_cer < best_cer:
            best_cer = val_cer
            best_epoch = epoch
            epochs_since_best = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_cer": val_cer,
                "val_wer": val_wer,
                "val_em": val_em,
                "vocab": tokenizer.vocab,
                "num_classes": num_classes,
                "ssl_initialized": ssl_loaded,
                "include_math": include_math,
                "domain_trained": domain,
                "width_reduction": WIDTH_REDUCTION,
            }, best_path)
            flag = " <- BEST SAVED"
        else:
            epochs_since_best += 1

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_cer": round(val_cer, 4),
            "val_wer": round(val_wer, 4),
            "val_em": round(val_em, 4),
            "per_domain": val_metrics["per_domain"],
            "lr": round(optimizer.param_groups[0]["lr"], 8),
            "time_sec": elapsed,
        })

        dom_str = " | ".join(
            f"{d[:4].upper()} CER: {m['cer']:.3f}" for d, m in val_metrics["per_domain"].items()
        )
        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train: {train_loss:.4f} | Val CER: {val_cer:.4f} | "
              f"WER: {val_wer:.4f} | EM: {val_em*100:.1f}% | {elapsed}s{flag}")
        if dom_str:
            print(f"     -> Domain Subsets: {dom_str}")

        if epochs_since_best >= patience:
            print(f"\n[*] Early stopping triggered at epoch {epoch} (patience={patience}).")
            break

    print(f"\n[OK] Primary CNN Model Training Complete! Best Val CER: {best_cer:.4f} at epoch {best_epoch}")
    print(f"     Checkpoint saved: {best_path}")

    return best_path, {
        "model": "CNN_BiLSTM_Attention",
        "best_cer": best_cer,
        "best_epoch": best_epoch,
        "checkpoint": best_path,
        "vocab_size": num_classes,
        "history": history,
    }


# =============================================================================
# 5. VISION TRANSFORMER (TrOCR) TRAINING WORKFLOW
# =============================================================================

def train_transformer_model(
    manifest: str,
    data_root: str,
    save_dir: str,
    model_name: str = "microsoft/trocr-small-printed",
    domain: str = "all",
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 5e-5,
    patience: int = 3,
    grad_accum: int = 2,
    include_math: bool = True,
    max_target_length: int = 128,
    max_samples: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[str, dict]:
    """Fine-tunes VisionEncoderDecoder (TrOCR) model across all specified domains."""
    print("\n" + "=" * 78)
    print(" [STAGE 2/2] FINE-TUNING VISION TRANSFORMER: TrOCR")
    print("=" * 78)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(save_dir, "trocr_finetuned")
    os.makedirs(out_dir, exist_ok=True)

    from transformers import VisionEncoderDecoderModel

    print(f"[*] Initializing TrOCR ({model_name})...")
    processor = load_trocr_processor_and_tokenizer(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    model = setup_trocr_model(model, processor)

    train_ds = DomainTrOCRDataset(
        processor=processor, manifest=manifest, split="train", data_root=data_root,
        domain_filter=domain, include_math=include_math, max_target_length=max_target_length,
        augment=True, max_samples=max_samples,
    )
    val_ds = DomainTrOCRDataset(
        processor=processor, manifest=manifest, split="valid", data_root=data_root,
        domain_filter=domain, include_math=include_math, max_target_length=max_target_length,
        augment=False, max_samples=max_samples if max_samples is None else max(8, max_samples // 2),
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"[!] Empty dataset partition for domain '{domain}'")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=trocr_collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=trocr_collate_fn, num_workers=0)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model backbone      : {model_name} ({n_params / 1e6:.1f}M parameters)")
    print(f"  Domain scope        : {domain.upper()} (historical, math, handwritten, printed)")
    print(f"  Train / Valid split : {len(train_ds)} / {len(val_ds)} samples")
    print(f"  Epochs / Batch / LR : {epochs} / {batch_size} / {lr} (grad accum: {grad_accum})")
    print(f"  Device              : {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = max(1, int(math.ceil(len(train_loader) / float(grad_accum))))
    total_steps = max(1, steps_per_epoch * epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1
    )

    print("\n[*] Evaluating Zero-Shot Baseline Performance before fine-tuning...")
    base_metrics = evaluate_transformer_split(model, processor, val_loader, device)
    base_cer = base_metrics["cer"]
    print(f"    Zero-Shot Baseline CER: {base_cer:.4f} | WER: {base_metrics['wer']:.4f} | EM: {base_metrics['exact_match']*100:.1f}%")
    for dom, m in base_metrics["per_domain"].items():
        print(f"      - {dom.capitalize():<14}: CER {m['cer']:.4f}, WER {m['wer']:.4f}, EM {m['em']*100:.1f}%")

    best_cer = base_cer
    best_epoch = 0
    since_best = 0
    history = [{
        "epoch": 0, "val_cer": round(base_cer, 4), "val_wer": round(base_metrics["wer"], 4),
        "val_em": round(base_metrics["exact_match"], 4), "per_domain": base_metrics["per_domain"],
        "note": "zero-shot baseline"
    }]

    print("\n[*] Commencing TrOCR Fine-Tuning...")
    for epoch in range(1, epochs + 1):
        model.train()
        start_t = time.time()
        running_loss, seen = 0.0, 0
        optimizer.zero_grad()

        for step, (pixel_values, labels, _, _) in enumerate(train_loader, 1):
            out = model(pixel_values=pixel_values.to(device), labels=labels.to(device))
            loss = out.loss / grad_accum
            loss.backward()

            if step % grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler.last_epoch < total_steps - 1:
                    scheduler.step()

            running_loss += out.loss.item() * pixel_values.size(0)
            seen += pixel_values.size(0)

        train_loss = running_loss / max(1, seen)
        val_metrics = evaluate_transformer_split(model, processor, val_loader, device)
        elapsed = round(time.time() - start_t, 1)

        val_cer = val_metrics["cer"]
        val_wer = val_metrics["wer"]
        val_em = val_metrics["exact_match"]

        flag = ""
        # Save whenever validation CER beats current best OR on first fine-tuning epoch if zero-shot wasn't fine-tuned
        if val_cer < best_cer or best_epoch == 0:
            best_cer = val_cer
            best_epoch = epoch
            since_best = 0
            model.save_pretrained(out_dir)
            processor.save_pretrained(out_dir)
            flag = " <- BEST SAVED"
        else:
            since_best += 1

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_cer": round(val_cer, 4),
            "val_wer": round(val_wer, 4),
            "val_em": round(val_em, 4),
            "per_domain": val_metrics["per_domain"],
            "time_sec": elapsed,
        })

        dom_str = " | ".join(
            f"{d[:4].upper()} CER: {m['cer']:.3f}" for d, m in val_metrics["per_domain"].items()
        )
        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val CER: {val_cer:.4f} | "
              f"WER: {val_wer:.4f} | EM: {val_em*100:.1f}% | {elapsed}s{flag}")
        if dom_str:
            print(f"     -> Domain Subsets: {dom_str}")

        if since_best >= patience:
            print(f"\n[*] Early stopping triggered at epoch {epoch} (patience={patience}).")
            break

    print(f"\n[OK] Vision Transformer Fine-Tuning Complete! Best Val CER: {best_cer:.4f} (epoch {best_epoch})")
    print(f"     Fine-tuned weights saved: {out_dir}")

    return out_dir, {
        "model": "Vision Transformer (TrOCR)",
        "base_cer": base_cer,
        "best_cer": best_cer,
        "best_epoch": best_epoch,
        "weights_dir": out_dir,
        "history": history,
    }


# =============================================================================
# 6. HELD-OUT TEST EVALUATION & BENCHMARK REPORT GENERATION
# =============================================================================

def run_held_out_benchmark(
    manifest: str,
    data_root: str,
    cnn_ckpt: Optional[str] = None,
    quantized_ckpt: Optional[str] = None,
    trocr_dir: Optional[str] = None,
    transformer_model: str = "microsoft/trocr-small-printed",
    domain: str = "all",
    max_samples: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Evaluates all trained architectures side-by-side on the held-out test split."""
    print("\n" + "=" * 78)
    print(" [EVALUATION] COMPREHENSIVE BENCHMARK ON HELD-OUT TEST SPLIT")
    print("=" * 78)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    # 1. Evaluate CNN (FP32)
    if cnn_ckpt and os.path.exists(cnn_ckpt):
        print(f"[*] Benchmarking Primary FP32 CNN ({cnn_ckpt})...")
        ckpt = torch.load(cnn_ckpt, map_location=device, weights_only=False)
        vocab = ckpt.get("vocab", DEFAULT_VOCAB)
        tokenizer = Tokenizer(vocab=vocab)
        model = CNN_BiLSTM_Attention(num_classes=len(tokenizer)).to(device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))

        test_ds = DomainOCRDataset(
            manifest=manifest, split="test", data_root=data_root, domain_filter=domain,
            include_math=True, exclude_infeasible=True, augment=False,
            tokenizer=tokenizer, max_samples=max_samples,
        )
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=domain_ocr_collate_fn)
        criterion = nn.CTCLoss(blank=tokenizer.blank_idx, zero_infinity=True)

        t0 = time.perf_counter()
        cnn_metrics = evaluate_cnn_split(model, test_loader, criterion, tokenizer, device)
        cnn_fps = len(test_ds) / max(0.001, (time.perf_counter() - t0))
        cnn_metrics["fps"] = cnn_fps
        results["cnn_fp32"] = cnn_metrics

    # 2. Evaluate Quantized CNN (INT8)
    if quantized_ckpt and os.path.exists(quantized_ckpt):
        print(f"[*] Benchmarking Edge Quantized INT8 CNN ({quantized_ckpt})...")
        q_ckpt = torch.load(quantized_ckpt, map_location="cpu", weights_only=False)
        vocab = q_ckpt.get("vocab", DEFAULT_VOCAB)
        tokenizer = Tokenizer(vocab=vocab)
        base_model = CNN_BiLSTM_Attention(num_classes=len(tokenizer))
        q_model = quantize_model(base_model)
        q_model.load_state_dict(q_ckpt["model_state_dict"])
        q_model.eval()

        test_ds = DomainOCRDataset(
            manifest=manifest, split="test", data_root=data_root, domain_filter=domain,
            include_math=True, exclude_infeasible=True, augment=False,
            tokenizer=tokenizer, max_samples=max_samples,
        )
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=domain_ocr_collate_fn)
        criterion = nn.CTCLoss(blank=tokenizer.blank_idx, zero_infinity=True)

        t0 = time.perf_counter()
        q_metrics = evaluate_cnn_split(q_model, test_loader, criterion, tokenizer, torch.device("cpu"))
        q_fps = len(test_ds) / max(0.001, (time.perf_counter() - t0))
        q_metrics["fps"] = q_fps
        results["cnn_int8"] = q_metrics

    # 3. Evaluate Transformer (TrOCR)
    has_weights = (
        trocr_dir
        and os.path.exists(trocr_dir)
        and os.path.exists(os.path.join(trocr_dir, "config.json"))
        and (
            os.path.exists(os.path.join(trocr_dir, "model.safetensors"))
            or os.path.exists(os.path.join(trocr_dir, "pytorch_model.bin"))
        )
    )
    trocr_load_path = trocr_dir if has_weights else transformer_model
    print(f"[*] Benchmarking Vision Transformer TrOCR ({trocr_load_path})...")
    try:
        from transformers import VisionEncoderDecoderModel
        processor = load_trocr_processor_and_tokenizer(trocr_load_path)
        trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_load_path).to(device)
        trocr_model = setup_trocr_model(trocr_model, processor)

        test_ds = DomainTrOCRDataset(
            processor=processor, manifest=manifest, split="test", data_root=data_root,
            domain_filter=domain, include_math=True, augment=False, max_samples=max_samples,
        )
        test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=trocr_collate_fn)

        t0 = time.perf_counter()
        trocr_metrics = evaluate_transformer_split(trocr_model, processor, test_loader, device)
        trocr_fps = len(test_ds) / max(0.001, (time.perf_counter() - t0))
        trocr_metrics["fps"] = trocr_fps
        results["trocr"] = trocr_metrics
    except Exception as exc:
        print(f"[!] Transformer evaluation failed: {exc}")

    # 4. Display Formatted Benchmark Comparison Table
    print("\n" + "=" * 78)
    print(" EMPIRICAL TEST BENCHMARK RESULTS (ACROSS TARGET DOMAINS)")
    print("=" * 78)
    header = f"{'Domain / Category':<22} | {'Primary CNN (FP32)':<18} | {'Edge CNN (INT8)':<18} | {'Vision Transformer':<18}"
    print(header)
    print("-" * 78)

    domains = ["historical", "mathematical", "handwritten", "printed", "OVERALL"]
    for dom in domains:
        cells = []
        for model_key in ["cnn_fp32", "cnn_int8", "trocr"]:
            if model_key not in results:
                cells.append("N/A")
                continue
            m = results[model_key]
            if dom == "OVERALL":
                cer_val = m.get("cer", 0.0)
                em_val = m.get("exact_match", 0.0)
                cells.append(f"CER {cer_val*100:.1f}% ({em_val*100:.0f}% EM)")
            else:
                p_dom = m.get("per_domain", {}).get(dom)
                if p_dom:
                    cells.append(f"CER {p_dom['cer']*100:.1f}%")
                else:
                    cells.append("-")

        dom_label = dom.capitalize() if dom != "OVERALL" else "OVERALL (ALL DATA)"
        print(f"{dom_label:<22} | {cells[0]:<18} | {cells[1]:<18} | {cells[2]:<18}")

    print("-" * 78)
    fps_cells = [
        f"{results[k]['fps']:.1f} FPS" if k in results else "N/A"
        for k in ["cnn_fp32", "cnn_int8", "trocr"]
    ]
    print(f"{'Throughput (FPS)':<22} | {fps_cells[0]:<18} | {fps_cells[1]:<18} | {fps_cells[2]:<18}")
    print("=" * 78)

    return results


# =============================================================================
# 7. MAIN ORCHESTRATOR
# =============================================================================

def train_all(
    manifest: Optional[str] = None,
    data_root: Optional[str] = None,
    save_dir: Optional[str] = None,
    model_choice: str = "both",
    domain: str = "all",
    cnn_epochs: int = 40,
    cnn_batch_size: int = 16,
    cnn_lr: float = 1e-3,
    cnn_patience: int = 10,
    cnn_use_ssl: bool = True,
    transformer_model: str = "microsoft/trocr-small-printed",
    transformer_epochs: int = 5,
    transformer_batch_size: int = 4,
    transformer_lr: float = 5e-5,
    transformer_grad_accum: int = 2,
    transformer_patience: int = 3,
    quantize: bool = True,
    evaluate_after: bool = True,
    max_samples: Optional[int] = None,
):
    """
    Unified training pipeline: trains both CNN and Transformer models on the entire
    dataset with domain-stratified handling for historical, math, and handwritten texts.
    """
    start_total = time.time()
    manifest = manifest or DEFAULT_MANIFEST
    data_root = data_root or os.path.join(PROJECT_ROOT, "data")
    save_dir = save_dir or os.path.join(PROJECT_ROOT, "src", "models", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        threads = max(1, (os.cpu_count() or 4) - 1)
        torch.set_num_threads(threads)
        print(f"[*] Training on {device} ({threads} CPU threads)")
    else:
        print(f"[*] Training on {device}")

    # 1. Scan and verify dataset distribution
    if not os.path.exists(manifest):
        print(f"[*] Manifest '{manifest}' not found. Generating standardized splits...")
        from src.make_splits import build_manifest
        build_manifest(data_root=data_root, out_csv=manifest)

    with open(manifest, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    domain_counts = Counter(classify_domain(r) for r in rows)
    split_counts = Counter(r.get("split", "") for r in rows)

    print("\n" + "=" * 78)
    print(" DATASET INVENTORY & DOMAIN DISTRIBUTION")
    print("=" * 78)
    print(f"  Total samples in manifest : {len(rows):,}")
    print(f"  Partitioning              : Train: {split_counts['train']}, Valid: {split_counts['valid']}, Test: {split_counts['test']}")
    print(f"  Domain Distribution       :")
    for dom, count in domain_counts.most_common():
        print(f"    * {dom.capitalize():<15}: {count:>5} samples ({100*count/len(rows):.1f}%)")
    print("=" * 78)

    model_choice = model_choice.lower()
    train_cnn = model_choice in ("both", "cnn", "all")
    train_trocr = model_choice in ("both", "transformer", "all")

    cnn_ckpt = None
    quantized_ckpt = None
    trocr_dir = None
    all_summary = {}

    # 2. Train Primary Architecture (CNN + BiLSTM + Attention)
    if train_cnn:
        cnn_ckpt, cnn_info = train_cnn_model(
            manifest=manifest,
            data_root=data_root,
            save_dir=save_dir,
            domain=domain,
            epochs=cnn_epochs,
            batch_size=cnn_batch_size,
            lr=cnn_lr,
            patience=cnn_patience,
            use_ssl=cnn_use_ssl,
            include_math=True,
            max_samples=max_samples,
            device=device,
        )
        all_summary["cnn"] = cnn_info

        # 3. Dynamic INT8 Quantization
        if quantize and cnn_ckpt and os.path.exists(cnn_ckpt):
            print("\n" + "=" * 78)
            print(" [QUANTIZATION] EXPORTING EDGE INT8 QUANTIZED MODEL")
            print("=" * 78)
            q_name = "cnn_bilstm_quantized.pth" if max_samples is None else "cnn_bilstm_quantized_smoke.pth"
            quantized_ckpt = os.path.join(save_dir, q_name)
            export_quantized_checkpoint(src_checkpoint=cnn_ckpt, dst_checkpoint=quantized_ckpt)
    else:
        # If CNN training was skipped, use existing checkpoint if present for benchmarking
        existing_cnn = os.path.join(save_dir, "cnn_bilstm_best.pth")
        if os.path.exists(existing_cnn):
            cnn_ckpt = existing_cnn
        existing_q = os.path.join(save_dir, "cnn_bilstm_quantized.pth")
        if os.path.exists(existing_q):
            quantized_ckpt = existing_q

    # 4. Train Transformer Architecture (Vision Transformer TrOCR)
    if train_trocr:
        trocr_dir, trocr_info = train_transformer_model(
            manifest=manifest,
            data_root=data_root,
            save_dir=save_dir,
            model_name=transformer_model,
            domain=domain,
            epochs=transformer_epochs,
            batch_size=transformer_batch_size,
            lr=transformer_lr,
            patience=transformer_patience,
            grad_accum=transformer_grad_accum,
            include_math=True,
            max_samples=max_samples,
            device=device,
        )
        all_summary["transformer"] = trocr_info
    else:
        existing_trocr = os.path.join(save_dir, "trocr_finetuned")
        if os.path.exists(existing_trocr):
            trocr_dir = existing_trocr

    # 5. Run Held-Out Test Split Benchmark & Comparison
    if evaluate_after:
        test_metrics = run_held_out_benchmark(
            manifest=manifest,
            data_root=data_root,
            cnn_ckpt=cnn_ckpt,
            quantized_ckpt=quantized_ckpt,
            trocr_dir=trocr_dir,
            transformer_model=transformer_model,
            domain=domain,
            max_samples=max_samples,
            device=device,
        )
        all_summary["test_benchmark"] = test_metrics

        # Save benchmark comparison CSV
        bench_csv_path = os.path.join(PROJECT_ROOT, "docs", "benchmark_train_all.csv")
        os.makedirs(os.path.dirname(bench_csv_path), exist_ok=True)
        with open(bench_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "domain", "cer", "wer", "exact_match", "fps"])
            for m_key, m_val in test_metrics.items():
                writer.writerow([m_key, "overall", m_val.get("cer"), m_val.get("wer"), m_val.get("exact_match", ""), m_val.get("fps")])
                for d_key, d_val in m_val.get("per_domain", {}).items():
                    writer.writerow([m_key, d_key, d_val.get("cer"), d_val.get("wer"), d_val.get("em", ""), ""])
        print(f"\n[OK] Benchmark CSV report exported to: {bench_csv_path}")

    # 6. Save Complete Training & Evaluation History
    history_json_path = os.path.join(save_dir, "train_all_history.json")
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2)

    total_time = round(time.time() - start_total, 1)
    print("\n" + "=" * 78)
    print(f" [ALL STAGES COMPLETE] Total pipeline execution time: {total_time}s")
    print(f"   Primary CNN Checkpoint       : {cnn_ckpt or 'Skipped'}")
    print(f"   Edge Quantized INT8 Model    : {quantized_ckpt or 'Skipped'}")
    print(f"   Vision Transformer Weights   : {trocr_dir or 'Skipped'}")
    print(f"   Training & Eval Log History  : {history_json_path}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="Train the entire dataset for both CNN and Transformer models across historical, mathematical, and handwritten texts with best accuracy."
    )
    parser.add_argument("--model", type=str, default="both", choices=["both", "cnn", "transformer"],
                        help="Which architecture to train: 'both' (default), 'cnn', or 'transformer'")
    parser.add_argument("--domain", type=str, default="all", choices=["all", "historical", "math", "handwritten", "printed"],
                        help="Target domain to train on (default: 'all' across entire dataset)")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to splits manifest CSV (default: data/splits.csv)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Data root directory")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to store checkpoints and history")
    parser.add_argument("--cnn-epochs", type=int, default=40,
                        help="Training epochs for CNN (default: 40)")
    parser.add_argument("--cnn-batch-size", type=int, default=16,
                        help="Batch size for CNN (default: 16)")
    parser.add_argument("--cnn-lr", type=float, default=1e-3,
                        help="Initial learning rate for CNN (default: 1e-3)")
    parser.add_argument("--cnn-patience", type=int, default=10,
                        help="Early stopping patience for CNN (default: 10)")
    parser.add_argument("--no-ssl", dest="cnn_use_ssl", action="store_false",
                        help="Disable SimCLR SSL pre-trained backbone transfer for CNN")
    parser.add_argument("--transformer-model", type=str, default="microsoft/trocr-small-printed",
                        help="TrOCR model variant (default: 'microsoft/trocr-small-printed'; or 'microsoft/trocr-base-handwritten')")
    parser.add_argument("--transformer-epochs", type=int, default=5,
                        help="Fine-tuning epochs for Transformer (default: 5)")
    parser.add_argument("--transformer-batch-size", type=int, default=4,
                        help="Batch size for Transformer (default: 4)")
    parser.add_argument("--transformer-lr", type=float, default=5e-5,
                        help="Learning rate for Transformer (default: 5e-5)")
    parser.add_argument("--transformer-grad-accum", type=int, default=2,
                        help="Gradient accumulation steps for Transformer (default: 2)")
    parser.add_argument("--transformer-patience", type=int, default=3,
                        help="Early stopping patience for Transformer (default: 3)")
    parser.add_argument("--no-quantize", dest="quantize", action="store_false",
                        help="Skip post-training dynamic INT8 quantization")
    parser.add_argument("--no-evaluate", dest="evaluate_after", action="store_false",
                        help="Skip held-out test split evaluation")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap sample count for fast smoke testing")

    args = parser.parse_args()

    train_all(
        manifest=args.manifest,
        data_root=args.data_root,
        save_dir=args.save_dir,
        model_choice=args.model,
        domain=args.domain,
        cnn_epochs=args.cnn_epochs,
        cnn_batch_size=args.cnn_batch_size,
        cnn_lr=args.cnn_lr,
        cnn_patience=args.cnn_patience,
        cnn_use_ssl=args.cnn_use_ssl,
        transformer_model=args.transformer_model,
        transformer_epochs=args.transformer_epochs,
        transformer_batch_size=args.transformer_batch_size,
        transformer_lr=args.transformer_lr,
        transformer_grad_accum=args.transformer_grad_accum,
        transformer_patience=args.transformer_patience,
        quantize=args.quantize,
        evaluate_after=args.evaluate_after,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
