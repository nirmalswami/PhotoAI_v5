import cv2

from detector import FaceDetector
from crop_engine import CropEngine

img = cv2.imread("input/Original.jpg")

if img is None:
    print("Image not found")
    exit()

detector = FaceDetector()

face = detector.detect(img)

if face is None:
    print("No face detected")
    exit()

engine = CropEngine()

result = engine.crop(img, face)

cv2.imwrite("test.jpg", result)

print("Done")