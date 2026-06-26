import cv2

from detector import FaceDetector

detector = FaceDetector()

img = cv2.imread("input/test.jpg")

result = detector.detect(img)

print(result)