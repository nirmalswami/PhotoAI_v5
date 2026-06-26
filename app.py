import os
import cv2
import time
import traceback

from detector import FaceDetector
from align import align_face
from crop_engine import CropEngine
from quality import QualityEngine

from config import (
    INPUT_DIR,
    OUTPUT_DIR,
    REJECTED_DIR,
    PROCESSED_FILE
)

# ---------------------------------------
# Create folders
# ---------------------------------------
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REJECTED_DIR, exist_ok=True)

# ---------------------------------------
# Initialize AI
# ---------------------------------------
print("=" * 60)
print("Loading AI Models...")
print("=" * 60)

detector = FaceDetector()
cropper = CropEngine()
quality = QualityEngine()

print("AI Ready\n")

# ---------------------------------------
# Load processed images
# ---------------------------------------
processed = set()

if os.path.exists(PROCESSED_FILE):
    with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
        processed = set(line.strip() for line in f if line.strip())

# ---------------------------------------
# Supported formats
# ---------------------------------------
extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

files = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(extensions)
]

total = len(files)

print("Total Images :", total)
print()

# ---------------------------------------
# Statistics
# ---------------------------------------
done = 0
rejected = 0
skipped = 0
failed = 0

start_time = time.time()

# ---------------------------------------
# Main Loop
# ---------------------------------------
for index, filename in enumerate(files, start=1):

    if filename in processed:
        skipped += 1
        continue

    try:
        path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(path)

        if img is None:
            print(f"[{index}/{total}] Cannot Read : {filename}")
            failed += 1
            continue

        # ----------------------------
        # Face Detection
        # ----------------------------
        result = detector.detect(img)

        if result is None:
            cv2.imwrite(os.path.join(REJECTED_DIR, filename), img)
            print(f"[{index}/{total}] No Face : {filename}")
            rejected += 1
            continue

        # ----------------------------
        # Multiple faces rejection
        # ----------------------------
        if result["face_count"] > 1:
            cv2.imwrite(os.path.join(REJECTED_DIR, filename), img)
            print(f"[{index}/{total}] Multiple Faces : {filename}")
            rejected += 1
            continue

        # ----------------------------
        # Align face
        # ----------------------------
        aligned, matrix = align_face(
            img,
            result["left_eye"],
            result["right_eye"]
        )

        # ----------------------------
        # Crop passport
        # ----------------------------
        passport = cropper.crop(aligned, result)

        # ----------------------------
        # Quality check (FIXED POSITION)
        # ----------------------------
        report = quality.check(passport, result)
        print(report)

        if not report["accepted"]:
            cv2.imwrite(os.path.join(REJECTED_DIR, filename), passport)
            print(filename, report["score"], report["reasons"])
            rejected += 1
            continue

        # ----------------------------
        # Save output
        # ----------------------------
        output_path = os.path.join(OUTPUT_DIR, filename)

        cv2.imwrite(
            output_path,
            passport,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        # ----------------------------
        # Save processed log
        # ----------------------------
        with open(PROCESSED_FILE, "a", encoding="utf-8") as f:
            f.write(filename + "\n")

        done += 1

        elapsed = time.time() - start_time
        avg = elapsed / done
        remaining = total - index
        eta = int(avg * remaining)

        print(f"[{index}/{total}] {filename} | Done | ETA {eta}s")

    except Exception as e:
        failed += 1
        print("\n" + "=" * 60)
        print(filename)
        print(e)
        traceback.print_exc()
        print("=" * 60 + "\n")

# ---------------------------------------
# Summary
# ---------------------------------------
elapsed = int(time.time() - start_time)

print("\n" + "=" * 60)
print("Completed")
print("=" * 60)

print("Processed :", done)
print("Rejected  :", rejected)
print("Skipped   :", skipped)
print("Failed    :", failed)

print("\nTime :", elapsed, "seconds")
print("=" * 60)