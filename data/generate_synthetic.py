import os
import random
import csv
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SAMPLE_TEXTS = [
    "Efficient Text Recognition",
    "Self Supervised Learning",
    "Deep Learning Major Project",
    "Convolutional Neural Network",
    "Bidirectional LSTM Model",
    "Attention Mechanism OCR",
    "A . B = A & B + C^2",
    "E = m * c ^ 2",
    "OCR System 2026",
    "Pattern Recognition Lab",
    "PyTorch Deep Learning",
    "Computer Vision Module",
    "Document Image Analysis",
    "Feature Extraction ResNet",
    "Sequence Sequence Model",
    "Information Science Dept",
    "NMAMIT Nitte Campus",
    "Synthetic Training Data",
    "Character Error Rate CER",
    "Word Error Rate WER",
    "1234567890 Math Sample",
    "x^2 + y^2 = z^2",
    "Data Science Research",
    "Machine Learning Pipeline"
]

def generate_synthetic_dataset(output_dir: str = "d:\\Major Project\\data\\synthetic", num_samples: int = 60):
    """
    Generates synthetic text image dataset with ground truth annotations.csv for training & testing.
    """
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "annotations.csv")

    records = []

    for i in range(num_samples):
        text = random.choice(SAMPLE_TEXTS)
        if random.random() > 0.5:
            # Add subtle variations
            text = text + " " + str(random.randint(1, 99))
        
        img_w, img_h = 512, 64
        image = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Position text with subtle offset
        offset_x = random.randint(10, 30)
        offset_y = random.randint(10, 18)
        draw.text((offset_x, offset_y), text, fill=(random.randint(0, 30), random.randint(0, 30), random.randint(0, 30)))
        
        img_np = np.array(image)
        
        # Apply synthetic noise/degradation variations
        if random.random() > 0.4:
            noise = np.random.normal(0, random.randint(5, 15), img_np.shape).astype(np.uint8)
            img_np = cv2.add(img_np, noise)
            
        if random.random() > 0.5:
            angle = random.uniform(-2.0, 2.0)
            M = cv2.getRotationMatrix2D((img_w // 2, img_h // 2), angle, 1.0)
            img_np = cv2.warpAffine(img_np, M, (img_w, img_h), borderValue=(255, 255, 255))
            
        filename = f"sample_{i:04d}.png"
        filepath = os.path.join(images_dir, filename)
        cv2.imwrite(filepath, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        
        # Store relative image path and text label
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
