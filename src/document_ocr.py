import os
import cv2
import numpy as np
import base64
from PIL import Image

from src.infer import OCRInferenceEngine

class LineSegmenter:
    """
    Line Segmentation Engine using Horizontal Projection Profiles and Morphological Operations.
    Splits full-page documents into individual text line crops with bounding boxes.
    """
    @staticmethod
    def segment_lines(img_np: np.ndarray, min_line_height: int = 12) -> list:
        """
        Returns list of dicts: [{'crop': np.ndarray, 'bbox': (x, y, w, h)}, ...]
        """
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY) if len(img_np.shape) == 3 else img_np.copy()
        h, w = gray.shape

        # Adaptive thresholding to isolate text strokes
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 9
        )

        # Horizontal dilation to connect adjacent characters into line strips
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)

        # Find line contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        line_boxes = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Filter small noise artifacts
            if bh >= min_line_height and bw > 30 and (bw * bh) > 400:
                line_boxes.append((x, y, bw, bh))

        # Sort line boxes top-to-bottom (by y coordinate)
        line_boxes = sorted(line_boxes, key=lambda b: b[1])

        results = []
        for (x, y, bw, bh) in line_boxes:
            # Add small padding
            pad_y = max(0, int(bh * 0.1))
            pad_x = max(0, int(bw * 0.05))

            y1 = max(0, y - pad_y)
            y2 = min(h, y + bh + pad_y)
            x1 = max(0, x - pad_x)
            x2 = min(w, x + bw + pad_x)

            line_crop = img_np[y1:y2, x1:x2]
            results.append({
                "crop": line_crop,
                "bbox": (x1, y1, x2 - x1, y2 - y1)
            })

        # Fallback if no contours found (treat whole image as single crop)
        if not results:
            results.append({
                "crop": img_np,
                "bbox": (0, 0, w, h)
            })

        return results


class DocumentOCREngine:
    """
    End-to-End Full Page Document & Multi-Line OCR Processing Engine.
    Supports separate execution of CNN-based model and Transformer-based model.
    """
    def __init__(self, inference_engine: OCRInferenceEngine = None):
        self.inference_engine = inference_engine if inference_engine is not None else OCRInferenceEngine()
        self.segmenter = LineSegmenter()

    def process_document(self, img_np: np.ndarray, model_type: str = "both") -> dict:
        line_segments = self.segmenter.segment_lines(img_np)
        
        annotated_img = img_np.copy()
        lines_output = []
        cnn_transcript = []
        trocr_transcript = []
        full_transcript = []

        model_type = model_type.lower()

        for idx, seg in enumerate(line_segments):
            crop = seg["crop"]
            x, y, bw, bh = seg["bbox"]

            res = self.inference_engine.run_pipeline(crop, model_type=model_type)

            cnn_text = res["cnn_bilstm_attention"]["predicted_text"] if res.get("cnn_bilstm_attention") else ""
            trocr_text = res["transformer_baseline"]["predicted_text"] if res.get("transformer_baseline") else ""

            if model_type == "cnn":
                line_pred = cnn_text
            elif model_type == "transformer":
                line_pred = trocr_text
            else:
                line_pred = f"[CNN]: {cnn_text} | [TrOCR]: {trocr_text}"

            if cnn_text: cnn_transcript.append(cnn_text)
            if trocr_text: trocr_transcript.append(trocr_text)
            full_transcript.append(line_pred)

            # Draw bounding box overlay on annotated image
            color = (34, 139, 34) if model_type == "cnn" else ((147, 20, 255) if model_type == "transformer" else (0, 242, 254))
            cv2.rectangle(annotated_img, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(
                annotated_img, f"Line {idx+1}", (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
            )

            lines_output.append({
                "line_num": idx + 1,
                "bbox": [x, y, bw, bh],
                "cnn_text": cnn_text,
                "trocr_text": trocr_text,
                "predicted_text": line_pred
            })

        # Encode annotated image to base64 for UI rendering
        _, buffer = cv2.imencode('.png', annotated_img)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "model_type": model_type,
            "total_lines_detected": len(line_segments),
            "cnn_transcript": "\n".join(cnn_transcript),
            "trocr_transcript": "\n".join(trocr_transcript),
            "full_transcript": "\n".join(full_transcript),
            "lines": lines_output,
            "annotated_image_base64": f"data:image/png;base64,{annotated_base64}"
        }


if __name__ == "__main__":
    # Self-test document OCR engine
    doc_engine = DocumentOCREngine()
    sample_path = "d:\\Major Project\\data\\samples\\printed_sample.png"
    if os.path.exists(sample_path):
        img = cv2.imread(sample_path)
        out = doc_engine.process_document(img, model_type="both")
        print(f"[OK] Document OCR Test passed! Detected {out['total_lines_detected']} line(s).")
        print(f"CNN Transcript:\n{out['cnn_transcript']}")
        print(f"TrOCR Transcript:\n{out['trocr_transcript']}")
