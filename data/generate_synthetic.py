import os
import random
import csv
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TTF_FONTS = ["arial.ttf", "calibri.ttf", "times.ttf", "consola.ttf", "segoeui.ttf"]

def get_font(name: str = "arial.ttf", size: int = 24):
    path = os.path.join("C:\\Windows\\Fonts", name)
    if os.path.exists(path):
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()

SAMPLE_TEXTS = [
    "Efficient Text Recognition System",
    "Self Supervised Learning 2026",
    "Deep Learning Major Project",
    "Convolutional Neural Network Architecture",
    "Bidirectional LSTM Sequence Model",
    "Attention Mechanism for OCR",
    "A . B = A & B + C^2",
    "E = m * c ^ 2",
    "Pattern Recognition Lab NMAMIT",
    "PyTorch Deep Learning Pipeline",
    "Computer Vision Image Processing",
    "Document Image Machine Translation",
    "Feature Map Extraction ResNet Backbone",
    "Connectionist Temporal Classification",
    "Information Science and Engineering",
    "Character and Word Error Rate CER",
    "x^2 + y^2 = z^2",
    "lim_{x->0} (sin x / x) = 1",
    "d/dx (e^x) = e^x",
    "a^2 + b^2 = c^2",
    "sum_{i=1}^n i = n(n+1)/2",
    "Optical Character Recognition Engine"
]

def generate_synthetic_dataset(output_dir: str = "d:\\Major Project\\data\\synthetic", num_samples: int = 150):
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "annotations.csv")

    records = []

    for i in range(num_samples):
        text = random.choice(SAMPLE_TEXTS)
        font_name = random.choice(TTF_FONTS)
        font = get_font(font_name, size=random.randint(22, 28))

        dummy_img = Image.new("RGB", (10, 10))
        dummy_draw = ImageDraw.Draw(dummy_img)
        bbox = dummy_draw.textbbox((0, 0), text, font=font)
        text_w = max(10, bbox[2] - bbox[0])
        text_h = max(10, bbox[3] - bbox[1])

        pad_x = random.randint(16, 24)
        pad_y = random.randint(8, 12)
        canvas_w = text_w + pad_x * 2
        canvas_h = text_h + pad_y * 2

        bg_val = random.randint(238, 255)
        image = Image.new("RGB", (canvas_w, canvas_h), color=(bg_val, bg_val, bg_val))
        draw = ImageDraw.Draw(image)
        draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font, fill=(random.randint(0, 30), random.randint(0, 30), random.randint(0, 30)))
        img_np = np.array(image)

        # Scale to standard height = 32
        h, w, _ = img_np.shape
        aspect = w / max(1, h)
        new_w = max(16, int(32 * aspect))
        resized = cv2.resize(img_np, (new_w, 32), interpolation=cv2.INTER_AREA)

        if random.random() > 0.5:
            noise = np.random.normal(0, random.randint(4, 10), resized.shape).astype(np.uint8)
            resized = cv2.add(resized, noise)

        filename = f"sample_{i:04d}.png"
        filepath = os.path.join(images_dir, filename)
        cv2.imwrite(filepath, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))

        rel_path = os.path.join("images", filename)
        records.append((rel_path, text))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(records)

    print(f"[OK] Generated {len(records)} synthetic dataset samples at '{output_dir}'. CSV saved to '{csv_path}'.")
    return csv_path

if __name__ == "__main__":
    generate_synthetic_dataset()
