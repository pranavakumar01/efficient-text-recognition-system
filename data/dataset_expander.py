import os
import random
import csv
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Category vocabulary pools
PRINTED_TEXTS = [
    "Efficient Text Recognition System",
    "Self Supervised Representation Learning",
    "Deep Learning Pattern Recognition",
    "Convolutional Neural Network Architecture",
    "Bidirectional Long Short Term Memory",
    "Bahdanau Sequence Attention Mechanism",
    "Information Science and Engineering",
    "NMAMIT Nitte Campus Research",
    "Optical Character Recognition Engine",
    "Feature Map Extraction ResNet Backbone",
    "Connectionist Temporal Classification Loss",
    "SimCLR Contrastive Pretraining Framework"
]

MATH_TEXTS = [
    "A . B = A & B + C^2",
    "E = m * c ^ 2",
    "f(x) = int_0^inf e^(-x^2) dx",
    "x^2 + y^2 = z^2",
    "lim_{x->0} (sin x / x) = 1",
    "d/dx (e^x) = e^x",
    "a^2 + b^2 = c^2",
    "sum_{i=1}^n i = n(n+1)/2",
    "P(A|B) = P(B|A)P(A) / P(B)",
    "det(A - lambda * I) = 0"
]

HANDWRITTEN_TEXTS = [
    "Handwritten Note Extraction Sample",
    "Quick Brown Fox Jumps Over Lazy Dog",
    "Lab Notebook Experiment Records 2026",
    "Deep Neural Nets for Sequence Modeling",
    "Computer Vision Image Processing Task",
    "Edge Device Deployment Optimization",
    "Character and Word Error Rate Metrics",
    "Low Resource Domain Adaptation Test"
]

HISTORICAL_DEGRADED_TEXTS = [
    "Historical Document Machine Translation",
    "Archival Manuscript Digitization Project",
    "Ancient Text Restoration and OCR",
    "Faded Ink and Paper Background Noise",
    "Binarization and Deskewing Enhancement",
    "Curved Text Line Sequence Recognition"
]

def generate_expanded_dataset(output_dir: str = "d:\\Major Project\\data\\expanded", num_samples: int = 500):
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "annotations.csv")

    records = []

    for i in range(num_samples):
        # Pick category randomly
        category = random.choice(["printed", "math", "handwritten", "historical"])
        
        if category == "printed":
            text = random.choice(PRINTED_TEXTS)
            font_size = random.randint(18, 24)
            is_noisy = False
            blur_level = 0
            skew_angle = random.uniform(-1.0, 1.0)
        elif category == "math":
            text = random.choice(MATH_TEXTS)
            font_size = random.randint(20, 26)
            is_noisy = False
            blur_level = 0
            skew_angle = 0.0
        elif category == "handwritten":
            text = random.choice(HANDWRITTEN_TEXTS)
            font_size = random.randint(16, 22)
            is_noisy = True
            blur_level = random.choice([0, 3])
            skew_angle = random.uniform(-2.5, 2.5)
        else: # historical
            text = random.choice(HISTORICAL_DEGRADED_TEXTS)
            font_size = random.randint(18, 22)
            is_noisy = True
            blur_level = random.choice([3, 5])
            skew_angle = random.uniform(-3.0, 3.0)

        # Render image
        img_w, img_h = 640, 64
        bg_color = random.randint(235, 255) if not is_noisy else random.randint(200, 230)
        image = Image.new("RGB", (img_w, img_h), color=(bg_color, bg_color, bg_color))
        draw = ImageDraw.Draw(image)
        
        offset_x = random.randint(10, 30)
        offset_y = random.randint(12, 20)
        text_color = (random.randint(0, 40), random.randint(0, 40), random.randint(0, 40))
        draw.text((offset_x, offset_y), text, fill=text_color)

        img_np = np.array(image)

        if blur_level > 0:
            img_np = cv2.GaussianBlur(img_np, (blur_level, blur_level), 0)

        if is_noisy:
            noise = np.random.normal(0, random.randint(10, 25), img_np.shape).astype(np.uint8)
            img_np = cv2.add(img_np, noise)

        if abs(skew_angle) > 0.1:
            M = cv2.getRotationMatrix2D((img_w // 2, img_h // 2), skew_angle, 1.0)
            img_np = cv2.warpAffine(img_np, M, (img_w, img_h), borderValue=(bg_color, bg_color, bg_color))

        filename = f"sample_{i:04d}.png"
        filepath = os.path.join(images_dir, filename)
        cv2.imwrite(filepath, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        rel_path = os.path.join("images", filename)
        records.append((rel_path, text, category))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "category"])
        for r in records:
            writer.writerow([r[0], r[1]])

    print(f"[OK] Generated {len(records)} expanded dataset samples at '{output_dir}'. CSV saved to '{csv_path}'.")
    return csv_path

if __name__ == "__main__":
    generate_expanded_dataset()
