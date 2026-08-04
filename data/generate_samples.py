import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_sample_images(output_dir: str = "d:\\Major Project\\data\\samples"):
    os.makedirs(output_dir, exist_ok=True)
    
    samples = [
        ("printed_sample.png", "Efficient Text Recognition System", False),
        ("handwritten_sample.png", "Self Supervised Learning 2026", True),
        ("math_equation_sample.png", "A . B = A & B + C^2", False),
        ("historical_document_sample.png", "Document Image Machine Translation", True)
    ]
    
    generated_paths = []
    
    for filename, text, is_noisy in samples:
        img_w, img_h = 600, 100
        image = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Draw text centered
        draw.text((30, 35), text, fill=(10, 10, 10))
        img_np = np.array(image)
        
        if is_noisy:
            # Add synthetic noise and degradation for historical / handwritten testing
            gauss_noise = np.random.normal(0, 15, img_np.shape).astype(np.uint8)
            img_np = cv2.add(img_np, gauss_noise)
            # Add subtle rotation/skew
            M = cv2.getRotationMatrix2D((img_w // 2, img_h // 2), 1.5, 1.0)
            img_np = cv2.warpAffine(img_np, M, (img_w, img_h), borderValue=(255, 255, 255))
        
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        generated_paths.append(filepath)
        
    print(f"Generated {len(generated_paths)} sample evaluation images in '{output_dir}'.")
    return generated_paths

if __name__ == "__main__":
    generate_sample_images()
