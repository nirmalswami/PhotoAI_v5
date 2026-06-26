import os
import csv
import cv2
import math
import numpy as np

from detector import FaceDetector
from config import INPUT_DIR

OUTPUT_DIR = "analysis"
CSV_FILE = os.path.join(OUTPUT_DIR, "dataset.csv")
REJECT_FILE = os.path.join(OUTPUT_DIR, "rejected.csv")
PROCESSED_FILE = "processed_analysis.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

detector = FaceDetector()

processed = set()

if os.path.exists(PROCESSED_FILE):
    with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
        processed = set(
            line.strip()
            for line in f
            if line.strip()
        )

if not os.path.exists(CSV_FILE):

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "File",
            "Image Width",
            "Image Height",
            "Faces",
            "Confidence",
            "Face Width",
            "Face Height",
            "Face Area",
            "Face %",
            "Eye Distance",
            "Mouth Distance",
            "Face Angle",
            "Eye Center X",
            "Eye Center Y",
            "Face Center X",
            "Face Center Y",
            "Blur",
            "Brightness"
        ])

if not os.path.exists(REJECT_FILE):

    with open(REJECT_FILE, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "File",
            "Reason"
        ])

files = [

    f for f in os.listdir(INPUT_DIR)

    if f.lower().endswith((
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp"
    ))
]

files = [

    f

    for f in files

    if f not in processed

]

total = len(files)

print("Pending :", total)

for index, file in enumerate(files, start=1):

    path = os.path.join(INPUT_DIR, file)

    img = cv2.imread(path)

    if img is None:

        with open(REJECT_FILE, "a", newline="", encoding="utf-8") as f:

            csv.writer(f).writerow([
                file,
                "Cannot Read"
            ])

        continue

    result = detector.detect(img)

    if result is None:

        with open(REJECT_FILE, "a", newline="", encoding="utf-8") as f:

            csv.writer(f).writerow([
                file,
                "No Face"
            ])

        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    brightness = np.mean(gray)

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:

        csv.writer(f).writerow([

            file,

            result["image_width"],
            result["image_height"],

            result["face_count"],

            round(result["confidence"],4),

            result["width"],
            result["height"],
            result["area"],
            result["face_percent"],

            round(result["eye_distance"],2),

            round(result["mouth_distance"],2),

            round(result["angle"],2),

            result["eye_center"][0],
            result["eye_center"][1],

            result["face_center"][0],
            result["face_center"][1],

            round(blur,2),

            round(brightness,2)

        ])

    with open(PROCESSED_FILE, "a", encoding="utf-8") as f:
        f.write(file + "\n")

    print(f"[{index}/{total}] {file}")

print("\nCompleted")