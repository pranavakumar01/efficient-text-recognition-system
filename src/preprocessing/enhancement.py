import cv2
import numpy as np
from PIL import Image

class ImagePreprocessor:
    """
    Preprocessing pipeline for document and scene text recognition.
    Includes contrast enhancement, noise reduction, deskewing, and binarization.
    """
    def __init__(self, target_height: int = 32, target_width: int = 128):
        self.target_height = target_height
        self.target_width = target_width

    def grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def denoise(self, image: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def binarize(self, image: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

    def deskew(self, image: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(image > 0))
        if coords.shape[0] == 0:
            return image
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def resize_and_pad(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        ratio = min(self.target_width / w, self.target_height / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        canvas = np.ones((self.target_height, self.target_width), dtype=np.uint8) * 255
        start_x = (self.target_width - new_w) // 2
        start_y = (self.target_height - new_h) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        return canvas

    def process(self, image: np.ndarray) -> dict:
        gray = self.grayscale(image)
        denoised = self.denoise(gray)
        enhanced = self.enhance_contrast(denoised)
        binarized = self.binarize(enhanced)
        deskewed = self.deskew(binarized)
        final_img = self.resize_and_pad(deskewed)

        return {
            "grayscale": gray,
            "denoised": denoised,
            "enhanced": enhanced,
            "binarized": binarized,
            "deskewed": deskewed,
            "final": final_img
        }
