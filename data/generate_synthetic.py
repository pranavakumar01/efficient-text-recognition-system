import os
import random
import csv
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

TTF_FONTS = [
    "arial.ttf", "arialbd.ttf", "calibri.ttf", "calibrib.ttf",
    "times.ttf", "timesbd.ttf", "consola.ttf", "consolab.ttf",
    "segoeui.ttf", "segoeuib.ttf", "georgia.ttf", "georgiab.ttf",
    "tahoma.ttf", "tahomabd.ttf", "trebuc.ttf", "trebucbd.ttf",
    "verdana.ttf", "verdanab.ttf", "cour.ttf", "courbd.ttf"
]

def get_font(name: str = "arial.ttf", size: int = 24):
    path = os.path.join("C:\\Windows\\Fonts", name)
    if os.path.exists(path):
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    arial_path = os.path.join("C:\\Windows\\Fonts", "arial.ttf")
    if os.path.exists(arial_path):
        return ImageFont.truetype(arial_path, size=size)
    return ImageFont.load_default()

DOMAINS = [
    "Efficient Text Recognition System",
    "Convolutional Neural Network Architecture",
    "Bidirectional Long Short Term Memory",
    "Bahdanau Sequence Attention Mechanism",
    "Connectionist Temporal Classification Loss",
    "Feature Map Extraction ResNet Backbone",
    "SimCLR Contrastive Representation Learning",
    "Vision Transformer Pretrained Decoder",
    "Optical Character Recognition Engine",
    "Deep Learning Pattern Recognition Lab",
    "Information Science and Engineering",
    "Multi Modal Document Processing Pipeline",
    "Autonomous Document Digitization System",
    "Adaptive Histogram Equalization Filtering",
    "Gradient Descent Backpropagation Optimizer",
    "Data Augmentation and Noise Invariance",
    "Character Error Rate and Word Error Rate",
    "Artificial Intelligence and Deep Learning",
    "Natural Language Processing Word Embeddings",
    "Low Latency Edge Device Deployment",
    "Document Image Machine Translation",
    "Supervised Fine Tuning on Labeled Data",
    "PyTorch Deep Learning Framework",
    "Computer Vision and Pattern Recognition",
    "Statistical Data Analysis and Findings",
    "Archival Manuscript Digitization Project",
    "Binarization and Deskewing Enhancement",
    "Century Old Heritage Print Preservation",
    "High Accuracy Handwritten Text Extraction",
    "Sequence to Sequence Attention Model",
    "Mathematical Equation Notation Parser",
    "Fast Greedy and Prefix Beam Search Decode",
    "Edge INT8 Quantization and Optimization",
    "Curved Text Line Sequence Recognition",
    "Multi Column Layout Analysis Engine",
    "Resolution Degradation and Ink Bleed Repair",
    "Residual Highway Connections in Deep CNNs",
    "Maximum Margin Temporal Loss Criterion",
    "Department of Information Science Archives",
    "NMAMIT Nitte Campus Research Project",
]

GENERAL_SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "Pack my box with five dozen liquor jugs",
    "How vexingly quick daft zebras jump",
    "Sphinx of black quartz judge my vow",
    "Two driven jocks help fax my big quiz",
    "Artificial intelligence continues to evolve rapidly",
    "Document analysis requires robust preprocessing steps",
    "High resolution scanning preserves subtle character strokes",
    "Neural networks recognize complex handwriting styles",
    "Modern computer vision pipelines process millions of images",
    "Digital preservation safeguards centuries of human knowledge",
    "Open source software accelerates scientific discovery",
    "Careful data preparation prevents overfitting during training",
    "Convolutional filters extract hierarchical visual features",
    "Bidirectional recurrent units capture past and future context",
    "Attention mechanisms dynamically focus on key image regions",
    "Quantization reduces memory footprint without losing accuracy",
    "Benchmarking on held out test sets gives honest metrics",
    "Preprocessed images have standardized dimensions and contrast",
    "Bilateral filtering removes noise while keeping edges sharp",
    "Contrast enhancement reveals faint ink traces on old paper",
    "Proper learning rate scheduling ensures stable convergence",
    "Regularization techniques prevent models from memorizing data",
    "Optical character recognition transforms paper into searchable text",
    "Fast inference allows real time text transcription on mobile",
    "Automated document processing reduces manual data entry errors",
    "Deep learning models achieve state of the art results",
    "Edge deployment brings machine intelligence to low power chips",
    "Multimodal models understand both visual and textual information",
    "Data science empowers evidence based decision making everywhere",
    "Research papers are published and shared across global networks",
    "Standardized evaluation protocols guarantee reproducible science",
    "Image processing transforms raw sensor signals into insights",
    "Statistical methods validate experimental findings rigorously",
    "Software engineers design reliable and scalable architectures",
    "Continuous integration catches regressions before release",
    "User interfaces should be intuitive responsive and accessible",
    "Mathematical reasoning underpins modern computational algorithms",
]

ALPHANUMERIC_AND_NUMBERS = [
    "Version 2.0 released in March 2026",
    "Accuracy achieved: 98.75% across test sets",
    "Document ID: #REC-2026-9041",
    "Batch size 16 with learning rate 0.001",
    "Latency: 42.5 ms per single page scan",
    "Page 45 of 320 in technical report",
    "Parameters: 5.27 million (INT8 quantized: 2.6 MB)",
    "Published date: 11 September 2026",
    "Score: CER 0.042, WER 0.089, FPS 45.2",
    "Model trained on 2,500 curated text samples",
    "Invoice #INV-2026-0814 total: $450.00",
    "Phone contact: +1 (800) 555-0199",
    "Tracking code: 1Z-999-999-02-1234-5678",
    "Coordinates: 40.7128 N, 74.0060 W",
    "Standard ISO 9001 quality management certified",
    "Time elapsed: 14 minutes and 32 seconds",
    "Dimensions: 640 x 480 pixels at 300 DPI",
    "Memory allocation: 128 MB RAM, 15% CPU load",
    "Report generated at 11:45:00 UTC",
    "Error rate decreased from 0.802 to 0.035",
]

WORDS_POOL = [
    "network", "system", "learning", "model", "accuracy", "training",
    "image", "process", "result", "feature", "layer", "weight",
    "gradient", "loss", "metric", "sample", "dataset", "epoch",
    "batch", "vector", "matrix", "tensor", "output", "input",
    "filter", "stride", "padding", "kernel", "pooling", "activation",
    "linear", "sequence", "temporal", "spatial", "channel", "attention",
    "token", "vocab", "label", "target", "predict", "decode",
    "encode", "score", "align", "error", "rate", "word",
    "character", "letter", "symbol", "digit", "number", "text",
    "document", "paper", "manuscript", "archive", "library", "record",
    "report", "project", "design", "method", "analysis", "study",
    "research", "science", "engine", "platform", "device", "hardware",
    "memory", "speed", "latency", "power", "compute", "scale"
]

def make_random_phrase():
    length = random.randint(2, 5)
    words = random.sample(WORDS_POOL, length)
    mode = random.choice(["title", "lower", "upper_first"])
    if mode == "title":
        return " ".join(w.capitalize() for w in words)
    elif mode == "upper_first":
        s = " ".join(words)
        return s[0].upper() + s[1:]
    return " ".join(words)

def build_text_pool(num_samples: int):
    pool = []
    pool.extend(DOMAINS)
    pool.extend(GENERAL_SENTENCES)
    pool.extend(ALPHANUMERIC_AND_NUMBERS)

    for t in DOMAINS[:15]:
        pool.append(t.lower())
        pool.append(t.upper())

    while len(pool) < num_samples:
        r = random.random()
        if r < 0.4:
            pool.append(make_random_phrase())
        elif r < 0.7:
            base = random.choice(GENERAL_SENTENCES)
            words = base.split()
            if len(words) > 3:
                start = random.randint(0, len(words) - 3)
                end = random.randint(start + 2, len(words))
                sub = " ".join(words[start:end])
                pool.append(sub[0].upper() + sub[1:])
            else:
                pool.append(base)
        elif r < 0.85:
            val = random.randint(10, 9999)
            unit = random.choice(["samples", "pixels", "ms", "epochs", "MB", "items", "%", "units"])
            lbl = random.choice(["Count", "Total", "Measured", "Recorded", "Output", "Value"])
            pool.append(f"{lbl}: {val} {unit}")
        else:
            w1 = random.choice(WORDS_POOL).capitalize()
            w2 = random.choice(WORDS_POOL).capitalize()
            pool.append(f"{w1} and {w2}")

    random.shuffle(pool)
    return pool[:num_samples]

def generate_synthetic_dataset(output_dir: str = "d:\\Major Project\\data\\synthetic", num_samples: int = 1200):
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "annotations.csv")

    texts = build_text_pool(num_samples)
    records = []

    for i, text in enumerate(texts):
        font_name = random.choice(TTF_FONTS)
        font_size = random.randint(22, 28)
        font = get_font(font_name, size=font_size)

        dummy = Image.new("RGB", (10, 10))
        draw_dummy = ImageDraw.Draw(dummy)
        bbox = draw_dummy.textbbox((0, 0), text, font=font)
        tw = max(10, bbox[2] - bbox[0])
        th = max(10, bbox[3] - bbox[1])

        dups = sum(1 for a, b in zip(text, text[1:]) if a == b)
        min_timesteps_needed = len(text) + dups + 4
        min_w_32 = min_timesteps_needed * 4

        pad_x = random.randint(16, 24)
        pad_y = random.randint(8, 12)
        cw = tw + pad_x * 2
        ch = th + pad_y * 2

        bg_c = random.randint(240, 255)
        fg_c = random.randint(0, 35)

        image = Image.new("RGB", (cw, ch), color=(bg_c, bg_c, bg_c))
        draw = ImageDraw.Draw(image)
        draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font, fill=(fg_c, fg_c, fg_c))
        img_np = np.array(image)

        h, w = img_np.shape[:2]
        aspect = w / max(1, h)
        new_w = max(min_w_32, int(32 * aspect))
        resized = cv2.resize(img_np, (new_w, 32), interpolation=cv2.INTER_AREA)

        if new_w < min_w_32:
            canvas = np.ones((32, min_w_32, 3), dtype=np.uint8) * bg_c
            canvas[:, :new_w] = resized
            resized = canvas

        if random.random() > 0.4:
            noise = np.random.normal(0, random.uniform(2, 6), resized.shape).astype(np.int16)
            resized = np.clip(resized.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        if random.random() > 0.5:
            angle = random.uniform(-1.5, 1.5)
            h_r, w_r = resized.shape[:2]
            M = cv2.getRotationMatrix2D((w_r // 2, 16), angle, 1.0)
            resized = cv2.warpAffine(resized, M, (w_r, 32), borderValue=(bg_c, bg_c, bg_c))

        filename = f"sample_{i:04d}.png"
        filepath = os.path.join(images_dir, filename)
        cv2.imwrite(filepath, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))

        rel_path = os.path.join("images", filename)
        records.append((rel_path, text))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(records)

    print(f"[OK] Generated {len(records)} high-accuracy synthetic dataset samples at '{output_dir}'.")
    return csv_path

if __name__ == "__main__":
    generate_synthetic_dataset()
