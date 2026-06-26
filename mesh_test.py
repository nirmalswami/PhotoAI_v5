import cv2

from mesh_detector import MeshDetector

img = cv2.imread("input/test.jpg")

detector = MeshDetector()

points = detector.detect(img)

print("Landmarks:", len(points))

for x, y in points:
    cv2.circle(img, (x, y), 1, (0,255,0), -1)

cv2.imwrite("mesh.jpg", img)

print("Done")