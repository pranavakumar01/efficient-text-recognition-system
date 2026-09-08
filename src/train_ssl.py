import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

from src.models.ssl_backbone import SimCLR_SSL_Backbone, info_nce_loss

class SimCLRTransform:
    """
    Data augmentation pipeline creating dual contrastive views (x_i, x_j) for SSL pre-training.
    """
    def __init__(self, target_h: int = 32, target_w: int = 256):
        self.target_h = target_h
        self.target_w = target_w

    def augment(self, img_np: np.ndarray) -> torch.Tensor:
        h, w = img_np.shape[:2]
        
        # 1. Random Crop / Resize
        crop_ratio = np.random.uniform(0.8, 1.0)
        ch, cw = int(h * crop_ratio), int(w * crop_ratio)
        sy, sx = np.random.randint(0, h - ch + 1), np.random.randint(0, w - cw + 1)
        cropped = img_np[sy:sy+ch, sx:sx+cw]
        
        resized = cv2.resize(cropped, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)

        # 2. Random Gaussian Blur
        if np.random.rand() > 0.5:
            ksize = np.random.choice([3, 5])
            resized = cv2.GaussianBlur(resized, (ksize, ksize), 0)

        # 3. Random Brightness / Contrast Jitter
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.7, 1.3)
            beta = np.random.randint(-20, 20)
            resized = cv2.convertScaleAbs(resized, alpha=alpha, beta=beta)

        # Normalize to float tensor [1, H, W]
        tensor = torch.from_numpy(resized).float().unsqueeze(0) / 255.0
        return tensor

    def __call__(self, img_path: str):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.ones((self.target_h, self.target_w), dtype=np.uint8) * 255
            
        view1 = self.augment(img)
        view2 = self.augment(img)
        return view1, view2


class UnlabeledSSLDataset(Dataset):
    """
    Unlabeled image dataset for self-supervised pre-training. No text labels.

    Two ways to choose images:

    * manifest=<path to data/splits.csv> (preferred) - uses only the images in
      the given split. This matters: SimCLR is unsupervised, but running it over
      the test images still lets the encoder shape its features around them, and
      the resulting test CER is then no longer a clean held-out measurement.

    * data_dir=<folder> (legacy) - walks that folder plus the project's other
      data folders. Note that this includes test images, so a run using this
      path must be described as transductive, not as a held-out result.

    The previous version accepted `data_dir` but then always appended a
    hardcoded list of every data folder to the search, so `--data_dir` could
    only ever add images and never restrict them.
    """

    IMG_EXT = (".png", ".jpg", ".jpeg")

    def __init__(self, data_dir: str = None, transform: SimCLRTransform = None,
                 manifest: str = None, split: str = "train"):
        self.transform = transform or SimCLRTransform()
        self.image_paths = []
        self.source = ""

        if manifest:
            if not os.path.exists(manifest):
                raise FileNotFoundError(
                    f"Split manifest not found at {manifest}. "
                    f"Run: python -m src.make_splits"
                )
            import csv
            root = os.path.dirname(os.path.abspath(manifest))
            with open(manifest, "r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    if split and row.get("split", "").lower() != split.lower():
                        continue
                    rel = (row.get("image_path") or "").replace("\\", "/")
                    path = os.path.join(root, os.path.normpath(rel))
                    if os.path.exists(path):
                        self.image_paths.append(path)
            self.source = f"{manifest} (split={split})"
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_root = os.path.join(project_root, "data")
            dirs_to_search = [d for d in (
                data_dir,
                os.path.join(data_root, "mathwriting"),
                os.path.join(data_root, "expanded"),
                os.path.join(data_root, "kaggle_dataset"),
            ) if d]

            seen = set()
            for d in dirs_to_search:
                if not os.path.exists(d):
                    continue
                for dirpath, _, files in os.walk(d):
                    for fname in files:
                        if fname.lower().endswith(self.IMG_EXT):
                            p = os.path.join(dirpath, fname)
                            if p not in seen:      # data_dir may overlap the others
                                seen.add(p)
                                self.image_paths.append(p)
            self.source = f"directory walk ({len(dirs_to_search)} folders, ALL SPLITS)"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_p = self.image_paths[idx]
        v1, v2 = self.transform(img_p)
        return v1, v2


def train_ssl(
    data_dir: str = None,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-3,
    save_dir: str = None,
    manifest: str = None,
    split: str = "train",
):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = save_dir or os.path.join(project_root, "src", "models", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] SSL Pre-training on device: {device}")

    # 1. Dataset & DataLoader
    dataset = UnlabeledSSLDataset(data_dir=data_dir, manifest=manifest, split=split)
    if len(dataset) == 0:
        raise SystemExit(
            f"[!] No images found for SSL pre-training (source: {dataset.source}).\n"
            f"    Build the data and manifest first:\n"
            f"      python data\\generate_synthetic.py\n"
            f"      python -m src.make_splits"
        )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"[*] Unlabeled images: {len(dataset)}")
    print(f"[*] Source          : {dataset.source}")
    if not manifest:
        print("[!] No manifest given, so this includes TEST images. SimCLR is")
        print("    unsupervised, but the encoder still adapts to them, which makes the")
        print("    downstream test CER transductive rather than held out. Pass")
        print("    --manifest data\\splits.csv to restrict to the training split.")
    if batch_size < 64:
        print(f"[!] Batch size {batch_size} gives only {2 * batch_size - 2} negatives per anchor.")
        print("    InfoNCE needs many negatives to produce a useful signal; SimCLR used")
        print("    256-8192. Treat any gain from this stage as unproven at this batch size.")

    # 2. Model & InfoNCE Loss
    encoder = SimCLR_SSL_Backbone(feature_dim=128).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)

    best_loss = float("inf")
    ssl_checkpoint_path = os.path.join(save_dir, "ssl_backbone_best.pth")

    print("\n" + "=" * 60)
    print(" [*] STARTING SIMCLR SELF-SUPERVISED PRE-TRAINING")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        encoder.train()
        running_loss = 0.0

        for view1, view2 in dataloader:
            view1 = view1.to(device)
            view2 = view2.to(device)

            optimizer.zero_grad()
            z1 = encoder(view1)
            z2 = encoder(view2)

            loss = info_nce_loss(z1, z2, temperature=0.07)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * view1.size(0)

        epoch_loss = running_loss / max(1, len(dataset))
        elapsed_sec = round(time.time() - start_time, 2)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | SimCLR Contrastive Loss: {epoch_loss:.4f} | Time: {elapsed_sec}s")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": encoder.state_dict(),
                "contrastive_loss": best_loss
            }
            torch.save(checkpoint_data, ssl_checkpoint_path)
            print(f"  [+] Saved new SSL pre-trained backbone to '{ssl_checkpoint_path}' (Loss: {best_loss:.4f})")

    print("\n" + "=" * 60)
    print(f" [OK] SSL Pre-training completed! Pre-trained backbone: '{ssl_checkpoint_path}'")
    print("=" * 60)
    return ssl_checkpoint_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR Self-Supervised Pre-training")
    parser.add_argument("--data_dir", "--data-dir", dest="data_dir", type=str, default=None,
                        help="Legacy directory mode. Walks this folder plus the other data "
                             "folders, which includes TEST images. Prefer --manifest.")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Restrict pre-training to one split of data/splits.csv")
    parser.add_argument("--split", type=str, default="train",
                        help="Split to use when --manifest is given (default: train)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", "--save-dir", dest="save_dir", type=str, default=None)
    args = parser.parse_args()

    train_ssl(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        manifest=args.manifest,
        split=args.split,
    )
