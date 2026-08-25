import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def get_available_font(font_name: str, size: int = 26):
    windows_font_dir = "C:\\Windows\\Fonts"
    candidate_path = os.path.join(windows_font_dir, font_name)
    if os.path.exists(candidate_path):
        try:
            return ImageFont.truetype(candidate_path, size=size)
        except Exception:
            pass
    arial_path = os.path.join(windows_font_dir, "arial.ttf")
    if os.path.exists(arial_path):
        return ImageFont.truetype(arial_path, size=size)
    return ImageFont.load_default()

def generate_sample_images(output_dir: str = "d:\\Major Project\\data\\samples"):
    os.makedirs(output_dir, exist_ok=True)
    
    samples = [
        ("printed_sample.png", "Efficient Text Recognition System", "arial.ttf", False),
        ("handwritten_sample.png", "Self Supervised Learning 2026", "comic.ttf", True),
        ("math_equation_sample.png", "A . B = A & B + C^2", "times.ttf", False),
        ("mathwriting_derivative_sample.png", "\\dot{y} = \\frac{dy}{dt}", "times.ttf", False),
        ("mathwriting_integral_sample.png", "C_{n} = \\int_{0}^{4} x^{n} \\rho(x) dx", "times.ttf", False),
        ("historical_document_sample.png", "Document Image Machine Translation", "georgia.ttf", True)
    ]
    
    generated_paths = []
    mathwriting_images_dir = "d:\\Major Project\\data\\mathwriting\\images"

    for filename, text, font_name, is_noisy in samples:
        filepath = os.path.join(output_dir, filename)

        font = get_available_font(font_name, size=28)
        dummy_img = Image.new("RGB", (10, 10))
        dummy_draw = ImageDraw.Draw(dummy_img)
        bbox = dummy_draw.textbbox((0, 0), text, font=font)
        text_w = max(10, bbox[2] - bbox[0])
        text_h = max(10, bbox[3] - bbox[1])

        pad_x = 24
        pad_y = 12
        canvas_w = text_w + pad_x * 2
        canvas_h = text_h + pad_y * 2

        bg_val = 230 if is_noisy else 255
        image = Image.new("RGB", (canvas_w, canvas_h), color=(bg_val, bg_val, bg_val))
        draw = ImageDraw.Draw(image)
        draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font, fill=(10, 10, 10))
        img_np = np.array(image)
        
        # Resize to standard height = 32
        h, w, _ = img_np.shape
        aspect = w / max(1, h)
        new_w = max(16, int(32 * aspect))
        resized = cv2.resize(img_np, (new_w, 32), interpolation=cv2.INTER_AREA)

        if is_noisy:
            gauss_noise = np.random.normal(0, 10, resized.shape).astype(np.uint8)
            resized = cv2.add(resized, gauss_noise)
            M = cv2.getRotationMatrix2D((new_w // 2, 16), 1.0, 1.0)
            resized = cv2.warpAffine(resized, M, (new_w, 32), borderValue=(bg_val, bg_val, bg_val))
        
        cv2.imwrite(filepath, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
        generated_paths.append(filepath)
        
    print(f"Generated {len(generated_paths)} sample evaluation images in '{output_dir}'.")
    return generated_paths

if __name__ == "__main__":
    generate_sample_images()
