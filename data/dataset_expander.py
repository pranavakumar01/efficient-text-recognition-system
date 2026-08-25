import os
import random
import csv
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Candidate TrueType fonts available on Windows
TTF_FONT_NAMES = [
    "arial.ttf", "calibri.ttf", "times.ttf", "consola.ttf", 
    "segoeui.ttf", "georgia.ttf", "tahoma.ttf", "trebuc.ttf", "verdana.ttf"
]
HANDWRITING_FONTS = ["comic.ttf", "segoepb.ttf", "calibril.ttf", "georgiaz.ttf"]

def get_available_font(font_name: str, size: int = 24):
    windows_font_dir = "C:\\Windows\\Fonts"
    candidate_path = os.path.join(windows_font_dir, font_name)
    if os.path.exists(candidate_path):
        try:
            return ImageFont.truetype(candidate_path, size=size)
        except Exception:
            pass
    # Fallback to standard arial or default
    arial_path = os.path.join(windows_font_dir, "arial.ttf")
    if os.path.exists(arial_path):
        return ImageFont.truetype(arial_path, size=size)
    return ImageFont.load_default()

# Expanded category vocabulary pools
PRINTED_TEXTS = [
    "Efficient Text Recognition System",
    "Self Supervised Representation Learning",
    "Deep Learning Pattern Recognition",
    "Convolutional Neural Network Architecture",
    "Bidirectional Long Short Term Memory",
    "Bahdanau Sequence Attention Mechanism",
    "Information Science and Engineering",
    "NMAMIT Nitte Campus Research Project",
    "Optical Character Recognition Engine",
    "Feature Map Extraction ResNet Backbone",
    "Connectionist Temporal Classification Loss",
    "SimCLR Contrastive Pretraining Framework",
    "Vision Transformer Pretrained Decoder",
    "Multi Modal Document Processing Pipeline",
    "Automated Form Processing and Digitization",
    "Artificial Intelligence and Machine Learning",
    "Natural Language Processing Word Embeddings",
    "Low Latency Edge Device Deployment",
    "Character Error Rate and Word Error Rate",
    "Adaptive Histogram Equalization Filtering",
    "Gradient Descent Backpropagation Optimizer",
    "Data Augmentation and Noise Invariance"
]

MATH_TEXTS = [
    "E = m * c ^ 2",
    "f(x) = int_0^inf e^(-x^2) dx",
    "x^2 + y^2 = z^2",
    "lim_{x->0} (sin x / x) = 1",
    "d/dx (e^x) = e^x",
    "a^2 + b^2 = c^2",
    "sum_{i=1}^n i = n(n+1)/2",
    "P(A|B) = P(B|A)P(A) / P(B)",
    "det(A - lambda * I) = 0",
    "nabla x B = mu_0 * (J + epsilon_0 * dE/dt)",
    "int (1 / x) dx = ln|x| + C",
    "e^(i * pi) + 1 = 0",
    "frac{d}{dx} [sin(x)] = cos(x)",
    "sqrt{x^2 + y^2} <= |x| + |y|",
    "sigma(z) = 1 / (1 + e^(-z))",
    "L_{CTC} = - ln P(y | x)",
    "alpha_t = exp(e_t) / sum exp(e_j)"
]

HANDWRITTEN_TEXTS = [
    "Handwritten Note Extraction Sample",
    "Quick Brown Fox Jumps Over Lazy Dog",
    "Lab Notebook Experiment Records 2026",
    "Deep Neural Nets for Sequence Modeling",
    "Computer Vision Image Processing Task",
    "Edge Device Deployment Optimization",
    "Character and Word Error Rate Metrics",
    "Low Resource Domain Adaptation Test",
    "Autonomous Document Digitization System",
    "Research Meeting Notes on Transformer Models",
    "Statistical Data Analysis and Findings",
    "Supervised Fine Tuning on Labeled Data"
]

HISTORICAL_DEGRADED_TEXTS = [
    "Historical Document Machine Translation",
    "Archival Manuscript Digitization Project",
    "Ancient Text Restoration and OCR",
    "Faded Ink and Paper Background Noise",
    "Binarization and Deskewing Enhancement",
    "Curved Text Line Sequence Recognition",
    "Department of Information Science Archives",
    "Century Old Heritage Print Preservation",
    "Multi Column Layout Analysis and Extraction",
    "Resolution Degradation and Ink Bleed Repair"
]

def render_text_image(text: str, category: str = "printed", target_height: int = 32) -> np.ndarray:
    """
    Renders high-quality TrueType antialiased text with tight bounding box padding.
    """
    font_size = random.randint(22, 28)
    if category == "handwritten":
        font_name = random.choice(HANDWRITING_FONTS)
        font = get_available_font(font_name, size=font_size)
    elif category == "math":
        font = get_available_font("times.ttf", size=font_size + 2)
    else:
        font_name = random.choice(TTF_FONT_NAMES)
        font = get_available_font(font_name, size=font_size)

    # Measure exact text dimensions using getbbox
    dummy_img = Image.new("RGB", (10, 10))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_w = max(10, bbox[2] - bbox[0])
    text_h = max(10, bbox[3] - bbox[1])

    # Create padded canvas
    pad_x = random.randint(12, 24)
    pad_y = random.randint(8, 14)
    canvas_w = text_w + pad_x * 2
    canvas_h = text_h + pad_y * 2

    # Background color variation
    if category in ["historical", "handwritten"]:
        bg_val = random.randint(215, 245)
    else:
        bg_val = random.randint(240, 255)
    bg_color = (bg_val, bg_val, bg_val)

    img = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Text color (black to dark charcoal)
    text_val = random.randint(0, 35)
    text_color = (text_val, text_val, text_val)

    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font, fill=text_color)
    img_np = np.array(img)

    # Aspect ratio resizing to target height = 32
    h, w, _ = img_np.shape
    aspect = w / max(1, h)
    new_w = max(16, int(target_height * aspect))
    resized = cv2.resize(img_np, (new_w, target_height), interpolation=cv2.INTER_AREA)

    # Optional augmentations per category
    if category == "handwritten":
        if random.random() > 0.5:
            resized = cv2.GaussianBlur(resized, (3, 3), 0)
    elif category == "historical":
        noise = np.random.normal(0, random.randint(5, 15), resized.shape).astype(np.uint8)
        resized = cv2.add(resized, noise)
        if random.random() > 0.4:
            resized = cv2.GaussianBlur(resized, (3, 3), 0)

    # Slight random skew [-1.5, 1.5]
    if random.random() > 0.6:
        angle = random.uniform(-1.5, 1.5)
        M = cv2.getRotationMatrix2D((new_w // 2, target_height // 2), angle, 1.0)
        resized = cv2.warpAffine(resized, M, (new_w, target_height), borderValue=bg_color)

    return resized


def generate_expanded_dataset(output_dir: str = "d:\\Major Project\\data\\expanded", num_samples: int = 600):
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "annotations.csv")

    records = []

    for i in range(num_samples):
        # Pick category randomly
        category = random.choices(
            ["printed", "math", "handwritten", "historical"],
            weights=[0.40, 0.25, 0.20, 0.15]
        )[0]

        if category == "printed":
            text = random.choice(PRINTED_TEXTS)
        elif category == "math":
            text = random.choice(MATH_TEXTS)
        elif category == "handwritten":
            text = random.choice(HANDWRITTEN_TEXTS)
        else:
            text = random.choice(HISTORICAL_DEGRADED_TEXTS)

        img_np = render_text_image(text, category=category, target_height=32)

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

    print(f"[OK] Generated {len(records)} high-resolution TrueType dataset samples at '{output_dir}'. CSV saved to '{csv_path}'.")
    return csv_path

if __name__ == "__main__":
    generate_expanded_dataset()
