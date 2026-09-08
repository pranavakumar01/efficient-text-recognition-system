"""
Supervised CTC training for the CNN-BiLSTM text recogniser.

What changed from the previous version, and why each mattered
------------------------------------------------------------
* Splits come from data/splits.csv (see src/make_splits.py) instead of a fresh
  `random_split` per run. Before, the partition was different every run, so no
  two runs were comparable, and `evaluate.py` scored on data that training had
  already seen.

* `input_lengths` is per-sample and derived from real content width. Before it
  was `torch.full(fill_value=seq_len)` - the padded batch width for every
  sample - which let CTC align characters onto blank white padding.

* Validation CER decodes only the timesteps that contain signal. Decoding the
  padded region injects noise into the metric you are steering by.

* The best checkpoint is chosen on validation CER, not validation loss. CTC
  loss and CER are only loosely coupled, so the old criterion could keep a
  checkpoint whose CER had got worse.

* Samples whose label is longer than the available timesteps are excluded by
  default. With `zero_infinity=True` their loss silently becomes zero: no
  gradient, no warning, and - because a zero still enters the batch mean - a
  training loss curve that looks better than reality. On this dataset that was
  41% of all rows.

* Augmentation is applied to the training split only. Previously `augment`
  defaulted to False and was never passed by any caller, so the augmentation
  code was dead.

* The vocabulary is derived from the training labels rather than assuming all
  95 printable ASCII characters, and it is stored in the checkpoint so
  inference can rebuild an identically-shaped model.

Typical use
-----------
    python -m src.make_splits
    python -m src.train --epochs 60                  # text only (recommended)
    python -m src.train --epochs 60 --include-math    # keep LaTeX in
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import (
    DEFAULT_MANIFEST,
    WIDTH_REDUCTION,
    OCRDataset,
    Tokenizer,
    build_vocab,
    ocr_collate_fn,
)
from src.models.cnn_bilstm_att import CNN_BiLSTM_Attention
from src.utils.metrics import OCRMetrics


def decode_batch(logits, input_lengths, tokenizer):
    """
    Greedy CTC decode, honouring each sample's true number of timesteps.

    Decoding the full padded width would read predictions off blank padding
    and add noise to the metric.
    """
    preds = logits.argmax(dim=-1).cpu()
    out = []
    for i in range(preds.size(0)):
        valid = int(input_lengths[i])
        out.append(tokenizer.decode(preds[i, :valid].tolist()))
    return out


def evaluate_split(model, loader, criterion, tokenizer, device):
    """Return (mean_loss, mean_cer, mean_wer, n_samples, examples)."""
    model.eval()
    total_loss, n_loss = 0.0, 0
    cers, wers = [], []
    examples = []

    with torch.no_grad():
        for images, targets, target_lengths, label_texts, input_lengths in loader:
            images = images.to(device)
            logits, _ = model(images)

            seq_len = logits.size(1)
            # Never claim more timesteps than the network actually produced.
            clamped = input_lengths.clamp(max=seq_len)

            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
            loss = criterion(log_probs, targets.to(device),
                             clamped.to(device), target_lengths.to(device))
            if torch.isfinite(loss):
                total_loss += loss.item() * images.size(0)
                n_loss += images.size(0)

            for pred, truth in zip(decode_batch(logits, clamped, tokenizer), label_texts):
                cers.append(OCRMetrics.calculate_cer(truth, pred))
                wers.append(OCRMetrics.calculate_wer(truth, pred))
                if len(examples) < 6:
                    examples.append((truth, pred))

    n = max(1, len(cers))
    return (total_loss / max(1, n_loss), sum(cers) / n, sum(wers) / n, len(cers), examples)


def train_model(
    manifest: str = None,
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 1e-3,
    save_dir: str = None,
    use_ssl: bool = False,
    include_math: bool = False,
    include_infeasible: bool = False,
    augment: bool = True,
    patience: int = 12,
    warmup_steps: int = 300,
    num_workers: int = 0,
    data_root: str = None,
    max_samples: int = None,
):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest = manifest or DEFAULT_MANIFEST
    save_dir = save_dir or os.path.join(project_root, "src", "models", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        threads = max(1, (os.cpu_count() or 4) - 1)
        torch.set_num_threads(threads)
        print(f"[*] Training on {device} ({threads} CPU threads)")
    else:
        print(f"[*] Training on {device}")

    exclude_infeasible = not include_infeasible
    ds_kwargs = dict(
        manifest=manifest,
        data_root=data_root,
        include_math=include_math,
        exclude_infeasible=exclude_infeasible,
    )

    # ---- Vocabulary from the training labels actually in use -------------
    probe = OCRDataset(split="train", **ds_kwargs)
    if len(probe) == 0:
        raise SystemExit(
            "[!] Training split is empty. Run 'python -m src.make_splits' first, "
            "and check your --include-math / --include-infeasible flags."
        )
    vocab = build_vocab(probe.labels)
    tokenizer = Tokenizer(vocab=vocab)
    num_classes = len(tokenizer)

    train_dataset = OCRDataset(split="train", tokenizer=tokenizer, augment=augment, **ds_kwargs)
    val_dataset = OCRDataset(split="valid", tokenizer=tokenizer, augment=False, **ds_kwargs)

    if len(val_dataset) == 0:
        raise SystemExit("[!] Validation split is empty; cannot select a checkpoint.")

    # --max-samples is a smoke test: it exercises the real vocabulary, the real
    # collate function and the real CTC call on a handful of batches so that a
    # shape or length bug surfaces in under a minute instead of three hours
    # into a full run. The vocabulary is still built from the complete training
    # split above, so num_classes matches a real run.
    train_source, val_source = train_dataset, val_dataset
    if max_samples:
        from torch.utils.data import Subset
        train_source = Subset(train_dataset, range(min(max_samples, len(train_dataset))))
        val_source = Subset(val_dataset, range(min(max(4, max_samples // 4), len(val_dataset))))

    train_loader = DataLoader(train_source, batch_size=batch_size, shuffle=True,
                              collate_fn=ocr_collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_source, batch_size=batch_size, shuffle=False,
                            collate_fn=ocr_collate_fn, num_workers=num_workers)

    print()
    print("=" * 72)
    print(" TRAINING CONFIGURATION")
    print("=" * 72)
    print(f"  manifest            : {manifest}")
    print(f"  train / valid       : {len(train_source)} / {len(val_source)} samples")
    if max_samples:
        print(f"                        (SMOKE TEST: capped from {len(train_dataset)} / {len(val_dataset)})")
    print(f"  LaTeX math included : {include_math}")
    print(f"  CTC-infeasible rows : {'INCLUDED' if include_infeasible else 'excluded'}")
    print(f"  augmentation        : {'train split only' if augment else 'off'}")
    print(f"  vocabulary          : {num_classes} classes (incl. blank) from {len(vocab)} chars")
    print(f"  epochs / batch / lr : {epochs} / {batch_size} / {lr}")
    print(f"  checkpoint criterion: lowest validation CER")
    print(f"  early stopping      : {patience} epochs without CER improvement")
    if len(vocab) <= 110:
        print(f"  charset             : {vocab!r}")

    if include_infeasible:
        bad = sum(1 for m in train_dataset.meta if not m["ctc_feasible"])
        if bad:
            print(f"\n[!] {bad} training sample(s) have a label longer than the available")
            print("    timesteps. CTC cannot align them; zero_infinity will zero their loss,")
            print("    so they contribute no gradient while still deflating your loss curve.")

    model = CNN_BiLSTM_Attention(num_classes=num_classes).to(device)

    if use_ssl:
        ssl_path = os.path.join(save_dir, "ssl_backbone_best.pth")
        print(f"\n[*] Loading SSL backbone weights: {ssl_path}")
        ok = model.load_ssl_weights(ssl_path)
        if not ok:
            print("[!] SSL weight loading reported failure; continuing from random init.")

    criterion = nn.CTCLoss(blank=tokenizer.blank_idx, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = []
    best_cer = float("inf")
    best_epoch = 0
    epochs_since_best = 0
    best_path = os.path.join(save_dir, "cnn_bilstm_best.pth")
    if max_samples:
        # A smoke run must never overwrite a real checkpoint. Selecting the best
        # of a few batches over one epoch produces a useless model, and silently
        # replacing hours of training with it would be worse than a crash.
        best_path = os.path.join(save_dir, "cnn_bilstm_smoketest.pth")
        print(f"\n[*] Smoke test: checkpoint redirected to {os.path.basename(best_path)}")
        print("    so that cnn_bilstm_best.pth is left untouched.")
    global_step = 0

    print("\n" + "=" * 72)
    print(f" STARTING CTC TRAINING (SSL init: {use_ssl})")
    print("=" * 72)

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        running_loss, seen, skipped = 0.0, 0, 0
        total_batches = len(train_loader)

        for batch_idx, (images, targets, target_lengths, _, input_lengths) in enumerate(train_loader, 1):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            # Linear LR warmup: CTC is unstable in the first few hundred steps
            # at lr=1e-3, and a diverged start is hard to recover from.
            global_step += 1
            if warmup_steps and global_step <= warmup_steps:
                for group in optimizer.param_groups:
                    group["lr"] = lr * global_step / warmup_steps

            optimizer.zero_grad()
            logits, _ = model(images)                      # [B, T, C]
            seq_len = logits.size(1)
            clamped = input_lengths.clamp(max=seq_len).to(device)

            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)   # [T, B, C]
            loss = criterion(log_probs, targets, clamped, target_lengths)

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                seen += images.size(0)
            else:
                skipped += images.size(0)

            if batch_idx % 25 == 0 or batch_idx == total_batches:
                avg = running_loss / max(1, seen)
                print(f"  epoch {epoch}/{epochs}  batch {batch_idx}/{total_batches}  "
                      f"train loss {avg:.4f}", flush=True)

        train_loss = running_loss / max(1, seen)
        val_loss, val_cer, val_wer, n_val, examples = evaluate_split(
            model, val_loader, criterion, tokenizer, device
        )
        scheduler.step()
        elapsed = round(time.time() - start, 1)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_cer": round(val_cer, 4),
            "val_wer": round(val_wer, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
            "time_sec": elapsed,
        })

        flag = ""
        if val_cer < best_cer:
            best_cer, best_epoch, epochs_since_best = val_cer, epoch, 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_cer": val_cer,
                "val_wer": val_wer,
                "vocab": tokenizer.vocab,
                "num_classes": num_classes,
                "ssl_initialized": use_ssl,
                "include_math": include_math,
                "include_infeasible": include_infeasible,
                "width_reduction": WIDTH_REDUCTION,
                "preprocessing": "ImagePreprocessor.prepare",
            }, best_path)
            flag = "  <- best, saved"
        else:
            epochs_since_best += 1

        print(f"epoch {epoch:03d}/{epochs:03d} | train {train_loss:.4f} | val {val_loss:.4f} | "
              f"CER {val_cer:.4f} | WER {val_wer:.4f} | {elapsed}s{flag}")
        if skipped:
            print(f"           [!] {skipped} sample(s) produced a non-finite loss this epoch")

        if epoch == 1 or flag:
            print("           sample predictions (truth -> prediction):")
            for truth, pred in examples[:3]:
                print(f"             {truth[:48]!r} -> {pred[:48]!r}")

        if epochs_since_best >= patience:
            print(f"\n[*] No CER improvement for {patience} epochs; stopping early at epoch {epoch}.")
            break

    if tokenizer.oov_counts:
        print(f"\n[!] {tokenizer.oov_report()}")
        print("    These were dropped from CTC targets but still counted against CER.")

    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 72)
    print(f" DONE. Best validation CER {best_cer:.4f} at epoch {best_epoch}")
    print(f" Checkpoint : {best_path}")
    print(f" History    : {history_path}")
    print("=" * 72)
    print("\nNext: python -m src.evaluate --split test")
    print("This is the first number that means anything - the test split has never")
    print("been trained on. Expect it to be worse than the old 0.5348, which was")
    print("measured on training data with an EasyOCR fallback substituting outputs.")
    return best_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CNN-BiLSTM CTC OCR model")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Split manifest (default: data/splits.csv)")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", "--save-dir", dest="save_dir", type=str, default=None)
    parser.add_argument("--use_ssl", "--use-ssl", dest="use_ssl", action="store_true",
                        help="Initialise from the SimCLR SSL backbone")
    parser.add_argument("--include-math", action="store_true",
                        help="Keep LaTeX samples. CTC cannot represent 2D math layout, "
                             "and 76%% of these have labels longer than the available timesteps.")
    parser.add_argument("--include-infeasible", action="store_true",
                        help="Keep samples whose label exceeds available timesteps "
                             "(their loss is silently zeroed by zero_infinity)")
    parser.add_argument("--no-augment", dest="augment", action="store_false",
                        help="Disable training-split augmentation")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Smoke test: cap the number of training samples so a shape "
                             "or length bug surfaces in a minute. Writes to "
                             "cnn_bilstm_smoketest.pth, never the real checkpoint.")
    args = parser.parse_args()

    train_model(
        manifest=args.manifest,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        use_ssl=args.use_ssl,
        include_math=args.include_math,
        include_infeasible=args.include_infeasible,
        augment=args.augment,
        patience=args.patience,
        warmup_steps=args.warmup_steps,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
