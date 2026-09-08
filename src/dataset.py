"""
Dataset, tokenizer and collation for the CTC text recogniser.

Key behaviours that changed, and why
------------------------------------
1. Training now runs the SAME preprocessing as inference, via
   `ImagePreprocessor.prepare()`. Previously this file did greyscale plus a
   plain resize while inference added bilateral denoising, CLAHE and
   deskewing, so the model was always evaluated on an unseen pixel
   distribution.

2. `__getitem__` returns each sample's true content width. `ocr_collate_fn`
   passes those widths through so training can give CTC a per-sample
   `input_lengths` instead of the padded batch width. With the padded width,
   CTC was free to align real characters onto blank white padding.

3. Batches are padded to a multiple of 4. The CNN yields
   T = floor(W / 4) timesteps, so any width not divisible by 4 silently
   discarded up to 3 rightmost pixel columns.

4. Out-of-vocabulary characters are counted and reported rather than silently
   dropped. Dropping them shortened the CTC target while CER was still scored
   against the full label - a penalty that was impossible to see.

5. The old "no annotations.csv" fallback assigned the literal label
   "Sample Target Text" to every image found on disk. That silently produced a
   dataset of wrong labels. It now raises instead.
"""

import csv
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.enhancement import ImagePreprocessor

# Full printable ASCII. Kept as the default for backward compatibility, but
# prefer `build_vocab(labels)` - a smaller vocabulary means fewer confusable
# classes, and once LaTeX is excluded most of these symbols never appear.
DEFAULT_VOCAB = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

# Written by src/make_splits.py; the single source of truth for partitioning.
DEFAULT_MANIFEST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "splits.csv"
)

# The CNN halves width twice, so timesteps T = floor(W / WIDTH_REDUCTION).
WIDTH_REDUCTION = 4


def build_vocab(labels, include_common_punctuation: bool = True) -> str:
    """
    Derive a vocabulary from the labels actually present, sorted for
    determinism. Optionally ensures common English punctuation (. , ! ? - & : ; ' ")
    is supported so validation and test splits never drop valid characters.
    """
    chars = set()
    for label in labels:
        chars.update(label)
    if include_common_punctuation:
        chars.update(".,!?-&:;'\"")
    return "".join(sorted(chars))


class Tokenizer:
    """
    Text tokenizer for CTC. Index 0 is reserved for the CTC blank.

    Set strict=True to raise on out-of-vocabulary characters instead of
    dropping them. Either way `oov_counts` records what was skipped so the
    problem is visible.
    """

    def __init__(self, vocab: str = DEFAULT_VOCAB, strict: bool = False):
        self.vocab = vocab
        self.char2idx = {char: idx + 1 for idx, char in enumerate(vocab)}
        self.idx2char = {idx + 1: char for idx, char in enumerate(vocab)}
        self.blank_idx = 0
        self.strict = strict
        self.oov_counts = {}

    def __len__(self):
        return len(self.vocab) + 1  # including blank

    def encode(self, text: str) -> list:
        encoded = []
        for char in text:
            idx = self.char2idx.get(char)
            if idx is not None:
                encoded.append(idx)
                continue
            self.oov_counts[char] = self.oov_counts.get(char, 0) + 1
            if self.strict:
                raise ValueError(
                    f"Character {char!r} is not in the vocabulary. "
                    f"Rebuild the vocabulary with build_vocab(), or pass strict=False "
                    f"to skip it (which will understate your CER)."
                )
        return encoded

    def decode(self, indices: list) -> str:
        """Greedy CTC decode: drop blanks, then collapse runs of the same index."""
        decoded = []
        prev_idx = None
        for idx in indices:
            if idx != self.blank_idx and idx != prev_idx:
                char = self.idx2char.get(idx)
                if char is not None:
                    decoded.append(char)
            prev_idx = idx
        return "".join(decoded)

    def oov_report(self) -> str:
        if not self.oov_counts:
            return "No out-of-vocabulary characters encountered."
        total = sum(self.oov_counts.values())
        worst = sorted(self.oov_counts.items(), key=lambda kv: -kv[1])[:12]
        detail = ", ".join(f"{c!r}x{n}" for c, n in worst)
        return f"{total} out-of-vocabulary character(s) skipped. Most frequent: {detail}"


class OCRDataset(Dataset):
    """
    Text / equation image dataset.

    Two ways to construct it:

      Manifest mode (preferred) - reads data/splits.csv, which guarantees the
      same partition across training and evaluation:
          OCRDataset(manifest=..., split="train", augment=True)

      Legacy directory mode - reads <data_dir>/annotations.csv:
          OCRDataset(data_dir="data/expanded", split=None)

    Set augment=True for the training split only; augmenting validation or
    test data makes the metric noisy and optimistic.
    """

    def __init__(
        self,
        data_dir: str = None,
        csv_filename: str = "annotations.csv",
        target_h: int = 32,
        tokenizer: Tokenizer = None,
        split: str = None,
        augment: bool = False,
        manifest: str = None,
        data_root: str = None,
        include_math: bool = True,
        exclude_infeasible: bool = False,
        use_full_preprocessing: bool = True,
        max_width: int = 640,
    ):
        self.data_dir = data_dir
        self.target_h = target_h
        self.tokenizer = tokenizer or Tokenizer()
        self.split = split
        self.augment = augment
        self.use_full_preprocessing = use_full_preprocessing
        self.max_width = max_width
        self.preprocessor = ImagePreprocessor(target_height=target_h, default_max_width=max_width)

        # (image_path, label) pairs; evaluate.py indexes this directly.
        self.samples = []
        # Parallel metadata: dicts with is_math / ctc_feasible / timesteps.
        self.meta = []

        if manifest or (data_dir is None):
            self._load_from_manifest(
                manifest or DEFAULT_MANIFEST,
                data_root=data_root,
                include_math=include_math,
                exclude_infeasible=exclude_infeasible,
            )
        else:
            self._load_from_directory(data_dir, csv_filename)

    def _load_from_manifest(self, manifest_path, data_root, include_math, exclude_infeasible):
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Split manifest not found at {manifest_path}.\n"
                f"Generate it first:  python -m src.make_splits"
            )
        if data_root is None:
            data_root = os.path.dirname(os.path.abspath(manifest_path))

        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if self.split and row.get("split", "").lower() != self.split.lower():
                    continue
                is_math = row.get("is_math", "0") == "1"
                feasible = row.get("ctc_feasible", "1") == "1"
                if is_math and not include_math:
                    continue
                if exclude_infeasible and not feasible:
                    continue

                rel = (row.get("image_path") or "").replace("\\", "/")
                img_path = os.path.join(data_root, os.path.normpath(rel))
                if not os.path.exists(img_path):
                    continue
                self.samples.append((img_path, row["label"]))
                self.meta.append({
                    "dataset": row.get("dataset", ""),
                    "is_math": is_math,
                    "ctc_feasible": feasible,
                    "timesteps": int(row.get("timesteps") or 0),
                })

    def _load_from_directory(self, data_dir, csv_filename):
        csv_path = os.path.join(data_dir, csv_filename)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"No annotation file at {csv_path}. Refusing to guess labels.\n"
                f"(An earlier version of this code assigned every image the literal label "
                f"'Sample Target Text' here, which silently produced a dataset of wrong "
                f"labels and made every metric meaningless.)"
            )
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if self.split and row.get("split") and row["split"].lower() != self.split.lower():
                    continue
                label = row.get("label")
                if not label:
                    continue
                rel = (row.get("image_path") or "").replace("\\", "/")
                img_path = os.path.join(data_dir, os.path.normpath(rel))
                if os.path.exists(img_path):
                    self.samples.append((img_path, label))
                    self.meta.append({"dataset": os.path.basename(data_dir),
                                       "is_math": False, "ctc_feasible": True, "timesteps": 0})

    def __len__(self):
        return len(self.samples)

    @property
    def labels(self):
        return [label for _, label in self.samples]

    def augment_image(self, img: np.ndarray) -> np.ndarray:
        """Mild geometric / photometric jitter. Training split only."""
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-3.0, 3.0)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        if np.random.rand() > 0.7:
            img = cv2.GaussianBlur(img, (3, 3), 0.5)
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.9, 1.1)
            beta = np.random.uniform(-10, 10)
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
        return img

    def preprocess_image(self, img_path: str) -> tuple:
        """
        Returns (tensor [1, H, W] in [0, 1], content_width).

        content_width excludes trailing white padding, so callers can compute
        honest CTC input lengths.
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.ones((self.target_h, 128), dtype=np.uint8) * 255

        # Augment the raw image, before normalisation, so the enhancement
        # stages see realistically degraded input.
        if self.augment:
            img = self.augment_image(img)

        if self.use_full_preprocessing:
            final_img, content_width = self.preprocessor.prepare(img, max_width=self.max_width)
        else:
            h, w = img.shape[:2]
            new_w = max(16, min(int(self.target_h * (w / float(max(1, h)))), self.max_width))
            final_img = cv2.resize(img, (new_w, self.target_h), interpolation=cv2.INTER_AREA)
            content_width = new_w

        tensor = torch.from_numpy(final_img).float().unsqueeze(0) / 255.0
        return tensor, int(content_width)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img_tensor, content_width = self.preprocess_image(img_path)
        encoded_label = self.tokenizer.encode(label)
        return (
            img_tensor,
            torch.tensor(encoded_label, dtype=torch.long),
            label,
            content_width,
        )


def ocr_collate_fn(batch):
    """
    Pads image widths to the batch maximum, rounded up to a multiple of 4.

    Returns (images, targets, target_lengths, label_texts, input_lengths).

    `input_lengths` is derived from each sample's own content width, not the
    padded batch width. Feeding CTC the padded width lets it place characters
    in blank padding, which was capping achievable accuracy regardless of how
    long the model trained.
    """
    images, target_tensors, label_texts, content_widths = zip(*batch)

    h = images[0].shape[1]
    max_w = max(img.shape[2] for img in images)
    # Round up so no rightmost columns are dropped by the //4 downsampling.
    max_w = int(np.ceil(max_w / WIDTH_REDUCTION) * WIDTH_REDUCTION)

    padded_images = []
    for img in images:
        pad_w = max_w - img.shape[2]
        if pad_w > 0:
            # 1.0 == white, matching the preprocessor's padding colour.
            padding = torch.ones((1, h, pad_w), dtype=torch.float32)
            img = torch.cat([img, padding], dim=2)
        padded_images.append(img)

    images_batch = torch.stack(padded_images, dim=0)  # [B, 1, H, W]

    target_lengths = torch.tensor([len(t) for t in target_tensors], dtype=torch.long)
    targets = (torch.cat(target_tensors, dim=0) if len(target_tensors) > 0
               else torch.tensor([], dtype=torch.long))

    timesteps_available = max_w // WIDTH_REDUCTION
    input_lengths = torch.tensor(
        [max(1, min(timesteps_available, w // WIDTH_REDUCTION)) for w in content_widths],
        dtype=torch.long,
    )

    return images_batch, targets, target_lengths, label_texts, input_lengths


if __name__ == "__main__":
    tok = Tokenizer()
    text = "OCR Test 123"
    enc = tok.encode(text)
    dec = tok.decode(enc)
    print(f"Tokenizer self-test -> {text!r} | encoded: {enc} | decoded: {dec!r}")
    assert dec == text, "Tokenizer encode/decode mismatch!"
    print("[OK] Tokenizer round-trip passed.")

    small = Tokenizer(vocab="abc")
    small.encode("abcXYZ")
    print(f"[OK] OOV tracking: {small.oov_report()}")

    if os.path.exists(DEFAULT_MANIFEST):
        ds = OCRDataset(split="train", include_math=False, exclude_infeasible=True)
        print(f"[OK] Manifest train split (text only, feasible only): {len(ds)} samples")
        if len(ds):
            img, target, label, width = ds[0]
            print(f"     first sample: shape={tuple(img.shape)} content_width={width} label={label!r}")
    else:
        print(f"[!] No manifest at {DEFAULT_MANIFEST}; run: python -m src.make_splits")
