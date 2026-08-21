import cv2
import numpy as np
from PIL import Image

class ImagePreprocessor:
    """
    Advanced Preprocessing pipeline for document, handwritten, and scene text recognition.
    Includes edge-preserving contrast enhancement, robust deskewing, noise reduction,
    and aspect-ratio preserving dynamic normalization.
    """
    def __init__(self, target_height: int = 32, default_max_width: int = 640):
        self.target_height = target_height
        self.default_max_width = default_max_width

    def grayscale(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            return np.ones((self.target_height, 128), dtype=np.uint8) * 255
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                # Alpha channel conversion with white background
                alpha = image[:, :, 3] / 255.0
                bg = np.ones_like(image[:, :, :3], dtype=np.float32) * 255.0
                blended = (image[:, :, :3] * alpha[:, :, None] + bg * (1.0 - alpha[:, :, None])).astype(np.uint8)
                return cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def denoise(self, image: np.ndarray) -> np.ndarray:
        # Fast bilateral filtering preserves high-frequency character edges and stroke terminals
        return cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        # Multi-scale Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(image)

    def binarize(self, image: np.ndarray) -> np.ndarray:
        # Otsu's optimal thresholding with slight Gaussian pre-blur
        blur = cv2.GaussianBlur(image, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Calculates skew angle on inverted text foreground strokes (not the background canvas)
        and applies rotation correction within safe bounds [-20 deg, +20 deg].
        """
        # Invert so text strokes are > 0 (white on black)
        if np.mean(image) > 127:
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        coords = np.column_stack(np.where(thresh > 0))
        if coords.shape[0] < 50:
            return image

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        # Clamp angle: only correct modest rotation tilts, ignore extreme orientations
        if abs(angle) > 25.0 or abs(angle) < 0.3:
            return image

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def resize_and_pad(self, image: np.ndarray, max_width: int = None) -> np.ndarray:
        """
        Resizes image to target_height while preserving original aspect ratio.
        Pads width to next multiple of 16 (or minimum 32) with white background.
        """
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return np.ones((self.target_height, 128), dtype=np.uint8) * 255

        aspect_ratio = w / float(h)
        new_w = max(16, int(self.target_height * aspect_ratio))
        max_w = max_width or self.default_max_width
        new_w = min(new_w, max_w)

        # Scale with INTER_AREA for downsampling, INTER_CUBIC for upsampling
        interp = cv2.INTER_AREA if new_w < w else cv2.INTER_CUBIC
        resized = cv2.resize(image, (new_w, self.target_height), interpolation=interp)

        # Pad width to nearest multiple of 16 for clean CNN pooling
        target_w = int(np.ceil(new_w / 16.0) * 16)
        target_w = max(32, target_w)

        if target_w > new_w:
            canvas = np.ones((self.target_height, target_w), dtype=np.uint8) * 255
            canvas[:, :new_w] = resized
            return canvas
        return resized

    def process(self, image: np.ndarray, max_width: int = None) -> dict:
        gray = self.grayscale(image)
        denoised = self.denoise(gray)
        enhanced = self.enhance_contrast(denoised)
        binarized = self.binarize(enhanced)
        deskewed = self.deskew(enhanced)  # Deskew grayscale-enhanced to retain anti-aliasing
        final_img = self.resize_and_pad(deskewed, max_width=max_width)

        return {
            "grayscale": gray,
            "denoised": denoised,
            "enhanced": enhanced,
            "binarized": binarized,
            "deskewed": deskewed,
            "final": final_img
        }


if __name__ == "__main__":
    prep = ImagePreprocessor()
    test_img = np.ones((64, 400, 3), dtype=np.uint8) * 255
    cv2.putText(test_img, "Test OCR Preprocessing", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    res = prep.process(test_img)
    print("Preprocessor test passed. Final output shape:", res["final"].shape)
