import os
import zipfile
import csv
import io
import random
import cv2
import numpy as np

DOWNLOADS_DIR = r"C:\Users\Asus\Downloads"
PROJECT_DATA_DIR = r"d:\Major Project\data"
KAGGLE_DIR = os.path.join(PROJECT_DATA_DIR, "kaggle_dataset")

def process_handwritten_dataset(max_samples=1000):
    zip_path = os.path.join(DOWNLOADS_DIR, "archive (1).zip")
    if not os.path.exists(zip_path):
        print(f"[!] Archive not found: {zip_path}")
        return []

    print("[*] Extracting Handwritten Dataset samples...")
    output_img_dir = os.path.join(KAGGLE_DIR, "handwritten")
    os.makedirs(output_img_dir, exist_ok=True)

    records = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Read validation or train CSV
        csv_name = "written_name_validation_v2.csv"
        if csv_name in z.namelist():
            f = io.TextIOWrapper(z.open(csv_name), encoding="utf-8")
            reader = csv.reader(f)
            header = next(reader) # ['FILENAME', 'IDENTITY']
            
            all_rows = list(reader)
            # Filter out NaN or UNREADABLE labels
            valid_rows = [r for r in all_rows if len(r) == 2 and r[1] and r[1] != "UNREADABLE"]
            
            selected_rows = random.sample(valid_rows, min(max_samples, len(valid_rows)))
            
            for fname, identity in selected_rows:
                # Find image path in zip
                target_path_1 = f"validation_v2/validation/{fname}"
                target_path_2 = f"train_v2/train/{fname}"
                
                zip_img_path = target_path_1 if target_path_1 in z.namelist() else (target_path_2 if target_path_2 in z.namelist() else None)
                if zip_img_path:
                    img_data = z.read(zip_img_path)
                    out_filepath = os.path.join(output_img_dir, fname)
                    with open(out_filepath, 'wb') as out_f:
                        out_f.write(img_data)
                    
                    rel_path = os.path.join("handwritten", fname)
                    records.append((rel_path, identity.strip(), "handwritten"))
                    
    print(f"[OK] Processed {len(records)} handwritten text samples.")
    return records

def process_historical_dataset(max_samples=200):
    zip_path = os.path.join(DOWNLOADS_DIR, "archive.zip")
    if not os.path.exists(zip_path):
        print(f"[!] Archive not found: {zip_path}")
        return []

    print("[*] Extracting Historical Document samples...")
    output_img_dir = os.path.join(KAGGLE_DIR, "historical")
    os.makedirs(output_img_dir, exist_ok=True)

    records = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        img_files = [f for f in z.namelist() if (f.endswith('.png') or f.endswith('.jpg')) and not f.startswith('__MACOSX')]
        selected_files = random.sample(img_files, min(max_samples, len(img_files)))

        for idx, z_path in enumerate(selected_files):
            img_data = z.read(z_path)
            fname = f"hist_{idx:04d}.png"
            out_filepath = os.path.join(output_img_dir, fname)
            with open(out_filepath, 'wb') as out_f:
                out_f.write(img_data)

            rel_path = os.path.join("historical", fname)
            # Placeholder/Synthesized label for historical page crops
            label = f"Historical Manuscript Document Sample {idx+1}"
            records.append((rel_path, label, "historical"))

    print(f"[OK] Processed {len(records)} historical document samples.")
    return records

def main():
    os.makedirs(KAGGLE_DIR, exist_ok=True)
    all_records = []

    # 1. Handwritten Dataset
    hw_records = process_handwritten_dataset(max_samples=1000)
    all_records.extend(hw_records)

    # 2. Historical Dataset
    hist_records = process_historical_dataset(max_samples=150)
    all_records.extend(hist_records)

    # Save master annotations CSV
    csv_path = os.path.join(KAGGLE_DIR, "annotations.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "category"])
        for r in all_records:
            writer.writerow([r[0], r[1], r[2]])

    print(f"\n=======================================================")
    print(f"[SUCCESS] Total Kaggle Dataset Samples: {len(all_records)}")
    print(f"[SUCCESS] Annotations saved to: '{csv_path}'")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
