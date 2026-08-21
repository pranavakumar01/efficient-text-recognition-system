import os
import argparse
import csv
import glob
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def parse_inkml_file(file_path: str):
    """
    Parses an InkML file to extract the normalized LaTeX label and stroke coordinate traces.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        ns = {'ink': 'http://www.w3.org/2003/InkML'}

        # 1. Extract ground truth label
        norm_label = ""
        for ann in root.findall('ink:annotation', ns) or root.findall('annotation'):
            if ann.attrib.get('type') == 'normalizedLabel' and ann.text:
                norm_label = ann.text.strip()
                break
        if not norm_label:
            for ann in root.findall('ink:annotation', ns) or root.findall('annotation'):
                if ann.attrib.get('type') == 'label' and ann.text:
                    norm_label = ann.text.strip()
                    break

        # 2. Extract coordinate traces
        traces = []
        for tr in root.findall('ink:trace', ns) or root.findall('trace'):
            coords = []
            text = tr.text.strip() if tr.text else ""
            if not text:
                continue
            for pt in text.split(','):
                parts = pt.strip().split()
                if len(parts) >= 2:
                    try:
                        x, y = float(parts[0]), float(parts[1])
                        coords.append((x, y))
                    except ValueError:
                        continue
            if coords:
                traces.append(coords)

        return norm_label, traces
    except Exception as e:
        return None, []

def render_traces_to_image(traces, target_height=64, padding=12, stroke_thickness=2):
    """
    Renders 2D pen strokes onto a white background canvas with anti-aliasing.
    """
    if not traces:
        return None

    all_x = [p[0] for tr in traces for p in tr]
    all_y = [p[1] for tr in traces for p in tr]
    if not all_x or not all_y:
        return None

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    orig_w = max(1.0, max_x - min_x)
    orig_h = max(1.0, max_y - min_y)

    # Scale coordinates proportionally to fit target height
    drawable_h = max(16, target_height - 2 * padding)
    scale = drawable_h / orig_h
    drawable_w = int(orig_w * scale)
    target_width = max(drawable_w + 2 * padding, 64)

    # Create white canvas (grayscale)
    canvas = np.ones((target_height, target_width), dtype=np.uint8) * 255

    for tr in traces:
        pts = []
        for x, y in tr:
            px = int((x - min_x) * scale + padding)
            py = int((y - min_y) * scale + padding)
            pts.append([px, py])
        
        pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        if len(pts) == 1:
            cv2.circle(canvas, tuple(pts[0]), stroke_thickness, 0, -1, lineType=cv2.LINE_AA)
        else:
            cv2.polylines(canvas, [pts_np], isClosed=False, color=0, thickness=stroke_thickness, lineType=cv2.LINE_AA)

    return canvas

def process_single_inkml(args_tuple):
    file_path, split_name, images_output_dir, target_height, stroke_thickness = args_tuple
    norm_label, traces = parse_inkml_file(file_path)
    if not norm_label or not traces:
        return None

    canvas = render_traces_to_image(traces, target_height=target_height, stroke_thickness=stroke_thickness)
    if canvas is None:
        return None

    file_id = os.path.splitext(os.path.basename(file_path))[0]
    out_filename = f"{split_name}_{file_id}.png"
    out_filepath = os.path.join(images_output_dir, out_filename)
    
    cv2.imwrite(out_filepath, canvas)
    rel_path = os.path.join("images", out_filename)
    return (rel_path, norm_label, split_name, "mathwriting")

def convert_mathwriting_dataset(
    dataset_dir: str = r"D:\mathwriting-2024",
    output_dir: str = r"d:\Major Project\data\mathwriting",
    max_train: int = 1500,
    max_val: int = 300,
    max_test: int = 200,
    target_height: int = 64,
    stroke_thickness: int = 2,
    num_workers: int = 8
):
    print("=" * 70)
    print(f"[*] MATHWRITING-2024 DATASET RASTERIZER & CONVERTER")
    print(f"[*] Source: '{dataset_dir}'")
    print(f"[*] Output: '{output_dir}'")
    print("=" * 70)

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found.")

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    splits_config = [
        ("train", max_train),
        ("valid", max_val),
        ("test", max_test)
    ]

    all_records = []

    for split_name, limit in splits_config:
        split_path = os.path.join(dataset_dir, split_name)
        if not os.path.exists(split_path):
            print(f"[!] Split directory '{split_name}' not found at {split_path}, skipping.")
            continue

        print(f"[*] Scanning '{split_name}' split (quota: {limit})...")
        # List files using os.scandir for high performance on large folders
        files = []
        with os.scandir(split_path) as it:
            for entry in it:
                if entry.name.endswith(".inkml") and entry.is_file():
                    files.append(entry.path)
                    if len(files) >= limit:
                        break

        print(f"[*] Found {len(files)} files to render for '{split_name}'. Processing in parallel...")
        tasks = [(f, split_name, images_dir, target_height, stroke_thickness) for f in files]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_single_inkml, tasks))

        valid_results = [r for r in results if r is not None]
        all_records.extend(valid_results)
        print(f"[OK] Successfully rendered {len(valid_results)} samples for split '{split_name}'.")

    # Write master annotations CSV
    csv_path = os.path.join(output_dir, "annotations.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "split", "category"])
        for r in all_records:
            writer.writerow(r)

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Total Rendered Math Samples: {len(all_records)}")
    print(f"[SUCCESS] Annotations saved to: '{csv_path}'")
    print("=" * 70)
    return csv_path

def main():
    parser = argparse.ArgumentParser(description="Convert MathWriting-2024 InkML files into rendered image dataset.")
    parser.add_argument("--dataset_dir", type=str, default=r"D:\mathwriting-2024", help="Path to MathWriting dataset root")
    parser.add_argument("--output_dir", type=str, default=r"d:\Major Project\data\mathwriting", help="Path to output directory")
    parser.add_argument("--max_train", type=int, default=1500, help="Maximum train samples to convert")
    parser.add_argument("--max_val", type=int, default=300, help="Maximum val samples to convert")
    parser.add_argument("--max_test", type=int, default=200, help="Maximum test samples to convert")
    parser.add_argument("--target_height", type=int, default=64, help="Target image height")
    parser.add_argument("--stroke_width", type=int, default=2, help="Stroke thickness in pixels")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker threads")

    args = parser.parse_args()
    convert_mathwriting_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        target_height=args.target_height,
        stroke_thickness=args.stroke_width,
        num_workers=args.workers
    )

if __name__ == "__main__":
    main()
