import cv2
import numpy as np
from PIL import Image

class ImagePreprocessor:
    """
    Advanced Preprocessing pipeline for document, handwritten, mathematical,
    and historical text recognition.
    Includes domain detection, adaptive historical enhancement, edge-preserving
    contrast enhancement, robust deskewing, and aspect-ratio preserving normalization.
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
        return image.copy()

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

    def enhance_historical(self, image: np.ndarray) -> np.ndarray:
        """
        Specialized restoration for degraded historical archival documents:
        1. Background illumination correction (rolling ball / large median filter).
        2. Contrast normalization to eliminate aged yellow paper tone and bleed-through.
        3. Morphological stroke reconnection for faded antique letterforms.
        """
        gray = self.grayscale(image)
        # Background estimation via large morphological opening
        bg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, bg_kernel)
        background = cv2.GaussianBlur(background, (25, 25), 0)

        # Difference-based background normalization
        diff = cv2.absdiff(gray, background)
        normalized = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Invert so ink is dark on light background
        normalized = 255 - normalized

        # Subtle CLAHE pass for enhanced character edge contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized)

        # Morphological closing to reconnect faded ink stroke fragments
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        restored = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, close_kernel)
        return restored

    @staticmethod
    def detect_domain(image_np: np.ndarray) -> str:
        """
        Heuristic image domain detector: classifies crop as 'handwritten', 'math', or 'printed'.
        Analyzes stroke orientation entropy, horizontal line structures, and contour properties.
        """
        if image_np is None:
            return "printed"

        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_np.shape) == 3 else image_np
        h, w = gray.shape
        if h < 8 or w < 8:
            return "printed"

        # Binarize inverted
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 1. Math check via MathFormulaParser
        from src.utils.math_recognizer import MathFormulaParser
        if MathFormulaParser.has_math_visual_structure(image_np):
            return "math"

        # 2. Handwriting check: evaluate connected component stroke irregularity and cursive baseline linkage
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours and len(contours) >= 2:
            aspect_ratios = []
            stroke_areas = []
            for c in contours:
                _, _, cw, ch = cv2.boundingRect(c)
                if ch >= 6 and cw >= 4:
                    aspect_ratios.append(cw / float(ch))
                    stroke_areas.append(cv2.contourArea(c))

            if len(aspect_ratios) >= 3:
                ar_variance = float(np.var(aspect_ratios))
                # Real cursive handwriting has connected letters creating wide irregular components (aspect ratio > 2.5)
                max_ar = max(aspect_ratios)
                if ar_variance > 1.5 or (max_ar > 2.8 and ar_variance > 0.8):
                    return "handwritten"

        return "printed"

    def deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Calculates skew angle on inverted text foreground strokes
        and applies rotation correction within safe bounds [-25 deg, +25 deg].
        """
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

        if abs(angle) > 25.0 or abs(angle) < 0.3:
            return image

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def resize_and_pad(self, image: np.ndarray, max_width: int = None, return_content_width: bool = False):
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            blank = np.ones((self.target_height, 128), dtype=np.uint8) * 255
            return (blank, 128) if return_content_width else blank

        aspect_ratio = w / float(h)
        new_w = max(16, int(self.target_height * aspect_ratio))
        max_w = max_width or self.default_max_width
        new_w = min(new_w, max_w)

        interp = cv2.INTER_AREA if new_w < w else cv2.INTER_CUBIC
        resized = cv2.resize(image, (new_w, self.target_height), interpolation=interp)

        target_w = int(np.ceil(new_w / 16.0) * 16)
        target_w = max(32, target_w)

        if target_w > new_w:
            canvas = np.ones((self.target_height, target_w), dtype=np.uint8) * 255
            canvas[:, :new_w] = resized
            return (canvas, new_w) if return_content_width else canvas
        return (resized, new_w) if return_content_width else resized

    def prepare(self, image: np.ndarray, max_width: int = None, is_historical: bool = False) -> tuple:
        """
        Primary preprocessing entry point.
        Returns (final_uint8_image, content_width).
        """
        if is_historical:
            enhanced = self.enhance_historical(image)
        else:
            gray = self.grayscale(image)
            denoised = self.denoise(gray)
            enhanced = self.enhance_contrast(denoised)

        deskewed = self.deskew(enhanced)
        final_img, content_width = self.resize_and_pad(
            deskewed, max_width=max_width, return_content_width=True
        )
        return final_img, content_width

    def process(self, image: np.ndarray, max_width: int = None, is_historical: bool = False) -> dict:
        gray = self.grayscale(image)
        denoised = self.denoise(gray)
        enhanced = self.enhance_historical(image) if is_historical else self.enhance_contrast(denoised)
        binarized = self.binarize(enhanced)
        deskewed = self.deskew(enhanced)
        final_img, content_width = self.prepare(image, max_width=max_width, is_historical=is_historical)
        domain = self.detect_domain(image)

        return {
            "grayscale": gray,
            "denoised": denoised,
            "enhanced": enhanced,
            "binarized": binarized,
            "deskewed": deskewed,
            "final": final_img,
            "content_width": content_width,
            "detected_domain": domain,
        }
