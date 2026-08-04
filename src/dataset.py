import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image

DEFAULT_VOCAB = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/=()., "

class Tokenizer:
    """
    Text Tokenizer for CTC Loss & Sequence Recognition.
    Reserves index 0 for CTC BLANK token.
    """
    def __init__(self, vocab: str = DEFAULT_VOCAB):
        self.vocab = vocab
        self.char2idx = {char: idx + 1 for idx, char in enumerate(vocab)}
        self.idx2char = {idx + 1: char for idx, char in enumerate(vocab)}
        self.blank_idx = 0

    def __len__(self):
        return len(self.vocab) + 1  # Including blank token

    def encode(self, text: str) -> list:
        encoded = []
        for char in text:
            if char in self.char2idx:
                encoded.append(self.char2idx[char])
        return encoded

    def decode(self, indices: list) -> str:
        decoded = []
        prev_idx = None
        for idx in indices:
            if idx != self.blank_idx:
                if idx != prev_idx:  # Collapse repeating characters for CTC greedy decoding
                    if idx in self.idx2char:
                        decoded.append(self.idx2char[idx])
            prev_idx = idx
        return "".join(decoded)


class OCRDataset(Dataset):
    """
    PyTorch Dataset for text image OCR loading.
    Supports CSV annotations (image_path, label) or simple directory listing.
    """
    def __init__(self, data_dir: str, csv_filename: str = "annotations.csv", target_h: int = 32, tokenizer: Tokenizer = None):
        self.data_dir = data_dir
        self.target_h = target_h
        self.tokenizer = tokenizer or Tokenizer()
        self.samples = []

        csv_path = os.path.join(data_dir, csv_filename)
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_p = os.path.join(data_dir, row["image_path"])
                    if os.path.exists(img_p):
                        self.samples.append((img_p, row["label"]))
        else:
            # Fallback: scan images directory without annotations (for SSL / testing)
            images_dir = os.path.join(data_dir, "images")
            search_dir = images_dir if os.path.exists(images_dir) else data_dir
            for fname in os.listdir(search_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(search_dir, fname), "Sample Target Text"))

    def __len__(self):
        return len(self.samples)

    def preprocess_image(self, img_path: str) -> torch.Tensor:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Create blank dummy image if read fails
            img = np.ones((self.target_h, 128), dtype=np.uint8) * 255

        h, w = img.shape
        aspect_ratio = w / float(h)
        new_w = max(16, int(self.target_h * aspect_ratio))
        resized = cv2.resize(img, (new_w, self.target_h), interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1] tensor shape [1, H, W]
        tensor = torch.from_numpy(resized).float().unsqueeze(0) / 255.0
        return tensor

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img_tensor = self.preprocess_image(img_path)
        encoded_label = self.tokenizer.encode(label)
        return img_tensor, torch.tensor(encoded_label, dtype=torch.long), label


def ocr_collate_fn(batch):
    """
    Collate function to dynamic pad image sequence widths and format CTC target tensors.
    """
    images, target_tensors, label_texts = zip(*batch)

    # Find max width in batch
    max_w = max(img.shape[2] for img in images)
    # Target height
    h = images[0].shape[1]

    padded_images = []
    for img in images:
        curr_w = img.shape[2]
        if curr_w < max_w:
            pad_w = max_w - curr_w
            # Pad with 1.0 (white background normalized)
            padding = torch.ones((1, h, pad_w), dtype=torch.float32)
            img_padded = torch.cat([img, padding], dim=2)
        else:
            img_padded = img
        padded_images.append(img_padded)

    images_batch = torch.stack(padded_images, dim=0)  # [B, 1, H, W]

    # Combine targets for CTCLoss
    target_lengths = torch.tensor([len(t) for t in target_tensors], dtype=torch.long)
    targets = torch.cat(target_tensors, dim=0) if len(target_tensors) > 0 else torch.tensor([], dtype=torch.long)

    return images_batch, targets, target_lengths, label_texts


if __name__ == "__main__":
    # Self-test Tokenizer and Dataset
    tok = Tokenizer()
    text = "OCR Test 123"
    enc = tok.encode(text)
    dec = tok.decode(enc)
    print(f"Tokenizer Self-Test -> Text: '{text}' | Encoded: {enc} | Decoded: '{dec}'")
    assert dec == text, "Tokenizer encoding/decoding mismatch!"
    print("[OK] Tokenizer test passed.")
