"""
Fine-tune TrOCR (VisionEncoderDecoder) on this project's data.

Why this file was rewritten
---------------------------
The previous version did not train anything. Its "training loop" ran
`trocr.predict()` over the validation set once per epoch with no optimizer, no
loss, and no backward pass, so every epoch produced a bit-identical CER, and
the `torch.save` at the end wrote four scalars of metadata rather than any
weights. The fine-tuned TrOCR checkpoint it advertised never existed.

Three further bugs in that loop:

* `img_tensor, label_seq, _, gt_text = val_dataset[idx]` put the *label text*
  into `_` and the *content width* into `gt_text`. CER was therefore computed
  against a stringified integer such as "96" for every sample. Combined with
  the `"Recognized Text Extraction"` placeholder this is a complete
  explanation for the reported TrOCR CER of 1.032 - the metric was never
  comparing the prediction to the ground truth at all.

* It fed TrOCR the 32-pixel-tall preprocessed greyscale line, converted back
  to RGB and then upscaled to 384x384 by the processor. TrOCR's ViT encoder was
  pretrained on full-resolution crops; a 32px source upscaled 12x carries
  almost no stroke detail. This file now reads the original image from disk.

* `trocr_params = 62000000` in evaluate.py, and the same literal in infer.py,
  is the parameter count of `trocr-small-*`. The code loads `trocr-base-printed`,
  which is roughly 334M. Both now measure the model they actually loaded.

Why fine-tuning matters more here than any decoder trick
--------------------------------------------------------
`trocr-base-printed` has never seen LaTeX. It cannot emit `\\frac{dy}{dt}`
regardless of beam width or post-processing, so its CER on the mathwriting
subset is bounded near 1.0 out of the box. Fine-tuning is the only way an
encoder-decoder learns that output alphabet. Unlike the CRNN, TrOCR has no
monotonic-alignment constraint, so 2D math layout is representable for it -
this is the model that should carry the math half of the project.

Cost warning
------------
`trocr-base-*` is ~334M parameters. On CPU a single epoch over 2700 samples
takes hours. Use --model microsoft/trocr-small-printed (~62M) to iterate, or
--max-samples to smoke-test the loop before committing to a long run.

Usage
-----
    python -m src.make_splits
    python -m src.train_transformer --max-samples 40 --epochs 1     # smoke test
    python -m src.train_transformer --model microsoft/trocr-base-handwritten
"""

import argparse
import csv
import json
import os
import time

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.dataset import DEFAULT_MANIFEST
from src.utils.metrics import OCRMetrics


class TrOCRManifestDataset(Dataset):
    """
    Yields (pixel_values, label_ids, label_text) straight from the original
    image files.

    Deliberately does NOT use ImagePreprocessor: that pipeline exists to make
    32-pixel-tall single-channel lines for the CRNN. TrOCR's processor does its
    own resize and normalisation, and giving it the original RGB crop is the
    whole point.
    """

    def __init__(self, processor, manifest=None, split="train", data_root=None,
                 include_math=True, max_target_length=96, max_samples=None):
        self.processor = processor
        self.max_target_length = max_target_length
        manifest = manifest or DEFAULT_MANIFEST
        if not os.path.exists(manifest):
            raise FileNotFoundError(
                f"Split manifest not found at {manifest}. Run: python -m src.make_splits"
            )
        data_root = data_root or os.path.dirname(os.path.abspath(manifest))

        self.samples = []
        with open(manifest, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if split and row.get("split", "").lower() != split.lower():
                    continue
                if not include_math and row.get("is_math") == "1":
                    continue
                rel = (row.get("image_path") or "").replace("\\", "/")
                path = os.path.join(data_root, os.path.normpath(rel))
                if os.path.exists(path):
                    self.samples.append((path, row["label"]))
                if max_samples and len(self.samples) >= max_samples:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, text = self.samples[idx]
        image = Image.open(path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values[0]
        ids = self.processor.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=self.max_target_length,
        ).input_ids
        # Padding must not contribute to the loss.
        pad_id = self.processor.tokenizer.pad_token_id
        labels = [tok if tok != pad_id else -100 for tok in ids]
        return pixel_values, torch.tensor(labels, dtype=torch.long), text


def collate(batch):
    pixel_values = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    texts = [b[2] for b in batch]
    return pixel_values, labels, texts


@torch.no_grad()
def validate(model, processor, loader, device, max_new_tokens=96, num_beams=1):
    model.eval()
    cers, wers, examples = [], [], []
    for pixel_values, _, texts in loader:
        generated = model.generate(
            pixel_values.to(device), max_new_tokens=max_new_tokens, num_beams=num_beams,
        )
        preds = processor.batch_decode(generated, skip_special_tokens=True)
        for pred, truth in zip(preds, texts):
            pred = pred.strip()
            cers.append(OCRMetrics.calculate_cer(truth, pred))
            wers.append(OCRMetrics.calculate_wer(truth, pred))
            if len(examples) < 5:
                examples.append((truth, pred))
    n = max(1, len(cers))
    return sum(cers) / n, sum(wers) / n, examples


def train_transformer_model(
    model_name: str = "microsoft/trocr-base-printed",
    manifest: str = None,
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 5e-5,
    save_dir: str = None,
    include_math: bool = True,
    max_samples: int = None,
    max_target_length: int = 96,
    grad_accum: int = 1,
    num_workers: int = 0,
    patience: int = 3,
):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = save_dir or os.path.join(project_root, "src", "models", "checkpoints")
    out_dir = os.path.join(save_dir, "trocr_finetuned")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)

    # These must be set for the decoder to train; a missing
    # decoder_start_token_id raises, and a missing pad id trains on padding.
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    train_ds = TrOCRManifestDataset(processor, manifest, "train", include_math=include_math,
                                    max_target_length=max_target_length, max_samples=max_samples)
    val_ds = TrOCRManifestDataset(processor, manifest, "valid", include_math=include_math,
                                  max_target_length=max_target_length,
                                  max_samples=max_samples if max_samples is None else max(8, max_samples // 4))
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("[!] Empty train or valid split. Run: python -m src.make_splits")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate, num_workers=num_workers)

    n_params = sum(p.numel() for p in model.parameters())
    print("=" * 72)
    print(" TrOCR FINE-TUNING (this actually updates weights)")
    print("=" * 72)
    print(f"  model         : {model_name}  ({n_params / 1e6:.1f}M parameters)")
    print(f"  device        : {device}")
    print(f"  train / valid : {len(train_ds)} / {len(val_ds)}")
    print(f"  math included : {include_math}")
    print(f"  epochs / bs   : {epochs} / {batch_size} (grad accum {grad_accum})")
    print(f"  output        : {out_dir}")
    if device.type == "cpu" and n_params > 1e8:
        est = len(train_ds) / max(1, batch_size) * 6 / 60
        print(f"\n[!] {n_params / 1e6:.0f}M parameters on CPU. Rough estimate: "
              f"{est:.0f}+ minutes per epoch.")
        print("    Use --model microsoft/trocr-small-printed to iterate faster.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max(1, (len(train_loader) // grad_accum) * epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1
    )

    print("\n[*] Zero-shot baseline before any fine-tuning:")
    base_cer, base_wer, base_examples = validate(model, processor, val_loader, device)
    print(f"    CER {base_cer:.4f} | WER {base_wer:.4f}")
    for truth, pred in base_examples[:3]:
        print(f"      {truth[:44]!r} -> {pred[:44]!r}")

    history = [{"epoch": 0, "val_cer": round(base_cer, 4), "val_wer": round(base_wer, 4),
                "note": "zero-shot, before fine-tuning"}]
    best_cer = base_cer
    best_epoch = 0
    since_best = 0

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.time()
        running, seen = 0.0, 0
        optimizer.zero_grad()

        for step, (pixel_values, labels, _) in enumerate(train_loader, 1):
            out = model(pixel_values=pixel_values.to(device), labels=labels.to(device))
            loss = out.loss / grad_accum
            loss.backward()

            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler.last_epoch < total_steps - 1:
                    scheduler.step()

            running += out.loss.item() * pixel_values.size(0)
            seen += pixel_values.size(0)
            if step % 10 == 0 or step == len(train_loader):
                print(f"  epoch {epoch}/{epochs} step {step}/{len(train_loader)} "
                      f"loss {running / max(1, seen):.4f}", flush=True)

        train_loss = running / max(1, seen)
        val_cer, val_wer, examples = validate(model, processor, val_loader, device)
        elapsed = round(time.time() - start, 1)

        flag = ""
        if val_cer < best_cer:
            best_cer, best_epoch, since_best = val_cer, epoch, 0
            # save_pretrained writes real weights, unlike the metadata-only
            # torch.save the previous version did.
            model.save_pretrained(out_dir)
            processor.save_pretrained(out_dir)
            flag = "  <- best, saved"
        else:
            since_best += 1

        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "val_cer": round(val_cer, 4), "val_wer": round(val_wer, 4),
                        "time_sec": elapsed})
        print(f"epoch {epoch:02d}/{epochs:02d} | train {train_loss:.4f} | "
              f"CER {val_cer:.4f} | WER {val_wer:.4f} | {elapsed}s{flag}")
        for truth, pred in examples[:3]:
            print(f"      {truth[:44]!r} -> {pred[:44]!r}")

        if since_best >= patience:
            print(f"\n[*] No improvement for {patience} epochs; stopping at epoch {epoch}.")
            break

    with open(os.path.join(save_dir, "trocr_training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 72)
    print(f" zero-shot CER {base_cer:.4f}  ->  fine-tuned CER {best_cer:.4f} (epoch {best_epoch})")
    if best_epoch == 0:
        print(" Fine-tuning did not beat the zero-shot baseline; nothing was saved.")
        print(" Try a lower --lr, more --epochs, or check the sample predictions above.")
    else:
        print(f" Weights: {out_dir}")
        print(f" Evaluate: python -m src.evaluate --model_type transformer --trocr-model {out_dir}")
    print("=" * 72)
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on the project splits")
    parser.add_argument("--model", dest="model_name", type=str,
                        default="microsoft/trocr-base-printed",
                        help="Use microsoft/trocr-small-printed on CPU, "
                             "or microsoft/trocr-base-handwritten for handwriting")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save_dir", "--save-dir", dest="save_dir", type=str, default=None)
    parser.add_argument("--no-math", dest="include_math", action="store_false",
                        help="Exclude LaTeX samples (TrOCR is the model that CAN learn them, "
                             "so usually leave them in)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap samples for a quick smoke test of the loop")
    parser.add_argument("--max-target-length", type=int, default=96)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    train_transformer_model(
        model_name=args.model_name,
        manifest=args.manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        include_math=args.include_math,
        max_samples=args.max_samples,
        max_target_length=args.max_target_length,
        grad_accum=args.grad_accum,
        num_workers=args.num_workers,
        patience=args.patience,
    )
