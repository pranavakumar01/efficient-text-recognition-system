import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from src.dataset import Tokenizer, OCRDataset, ocr_collate_fn
from src.models.cnn_bilstm_att import CNN_BiLSTM_Attention
from src.utils.metrics import OCRMetrics

def train_model(
    data_dir: str = "d:\\Major Project\\data\\synthetic",
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    save_dir: str = "d:\\Major Project\\src\\models\\checkpoints",
    use_ssl: bool = False
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # 1. Dataset & Tokenizer Setup
    tokenizer = Tokenizer()
    num_classes = len(tokenizer)

    dataset_dirs = [data_dir, "d:\\Major Project\\data\\expanded", "d:\\Major Project\\data\\kaggle_dataset"]
    datasets = []
    for d in dataset_dirs:
        if os.path.exists(d):
            ds = OCRDataset(data_dir=d, tokenizer=tokenizer)
            if len(ds) > 0:
                datasets.append(ds)

    if datasets:
        from torch.utils.data import ConcatDataset
        full_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    else:
        from data.generate_synthetic import generate_synthetic_dataset
        generate_synthetic_dataset(output_dir=data_dir)
        full_dataset = OCRDataset(data_dir=data_dir, tokenizer=tokenizer)

    # Train / Val Split (80% train, 20% val)
    val_size = max(1, int(len(full_dataset) * 0.2))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=ocr_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=ocr_collate_fn)

    print(f"[*] Dataset total: {len(full_dataset)} | Train: {train_size} | Val: {val_size}")

    # 2. Model, Loss, Optimizer
    model = CNN_BiLSTM_Attention(num_classes=num_classes).to(device)

    # Load SSL pre-trained backbone weights if requested
    if use_ssl:
        ssl_path = os.path.join(save_dir, "ssl_backbone_best.pth")
        print(f"[*] Loading pre-trained SSL weights from: {ssl_path}")
        model.load_ssl_weights(ssl_path)

    criterion = nn.CTCLoss(blank=tokenizer.blank_idx, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    history = []
    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(save_dir, "cnn_bilstm_best.pth")

    print("\n" + "=" * 60)
    print(f" [*] STARTING SUPERVISED CTC TRAINING (SSL Init: {use_ssl})")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        running_train_loss = 0.0

        for images, targets, target_lengths, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            logits, _ = model(images)  # [B, Seq_Len, Num_Classes]

            b_size, seq_len, _ = logits.size()
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)  # [Seq_Len, B, Num_Classes]
            input_lengths = torch.full(size=(b_size,), fill_value=seq_len, dtype=torch.long, device=device)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                running_train_loss += loss.item() * b_size

        train_loss = running_train_loss / max(1, train_size)

        # Validation Loop
        model.eval()
        running_val_loss = 0.0
        cer_scores = []
        wer_scores = []

        with torch.no_grad():
            for images, targets, target_lengths, label_texts in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)

                logits, _ = model(images)
                b_size, seq_len, _ = logits.size()
                log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
                input_lengths = torch.full(size=(b_size,), fill_value=seq_len, dtype=torch.long, device=device)

                val_loss = criterion(log_probs, targets, input_lengths, target_lengths)
                if not torch.isnan(val_loss) and not torch.isinf(val_loss):
                    running_val_loss += val_loss.item() * b_size

                # Calculate CER and WER for validation samples
                preds = logits.argmax(dim=-1).cpu().numpy()
                for i, pred_seq in enumerate(preds):
                    decoded_pred = tokenizer.decode(pred_seq)
                    gt_text = label_texts[i]
                    cer = OCRMetrics.calculate_cer(gt_text, decoded_pred)
                    wer = OCRMetrics.calculate_wer(gt_text, decoded_pred)
                    cer_scores.append(cer)
                    wer_scores.append(wer)

        avg_val_loss = running_val_loss / max(1, val_size)
        avg_cer = sum(cer_scores) / max(1, len(cer_scores))
        avg_wer = sum(wer_scores) / max(1, len(wer_scores))
        elapsed_sec = round(time.time() - start_time, 2)

        scheduler.step(avg_val_loss)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_cer": round(avg_cer, 4),
            "val_wer": round(avg_wer, 4),
            "time_sec": elapsed_sec
        }
        history.append(epoch_record)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | CER: {avg_cer:.4f} | WER: {avg_wer:.4f} | Time: {elapsed_sec}s")

        # Save Best Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "val_cer": avg_cer,
                "val_wer": avg_wer,
                "vocab": tokenizer.vocab,
                "ssl_initialized": use_ssl
            }
            torch.save(checkpoint_data, best_checkpoint_path)
            print(f"  [+] Saved new best model checkpoint to '{best_checkpoint_path}' (Val Loss: {avg_val_loss:.4f})")

    # Save training history JSON
    history_path = os.path.join(save_dir, "training_history.json")
    import json
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"  [+] Saved training metrics history to '{history_path}'")

    print("\n" + "=" * 60)
    print(f" [OK] Training completed! Best Checkpoint: '{best_checkpoint_path}'")
    print("=" * 60)
    return best_checkpoint_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN-BiLSTM-Attention OCR Model")
    parser.add_argument("--data_dir", type=str, default="d:\\Major Project\\data\\synthetic")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="d:\\Major Project\\src\\models\\checkpoints")
    parser.add_argument("--use_ssl", action="store_true", help="Load pre-trained SimCLR SSL backbone weights before training")
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        use_ssl=args.use_ssl
    )
