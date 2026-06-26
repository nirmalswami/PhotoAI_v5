import cv2
import math
import numpy as np
from insightface.app import FaceAnalysis

from config import DETECTION_SIZE
from config import CPU_PROVIDER

print("Loading InsightFace Model...")

app = FaceAnalysis(
    providers=[CPU_PROVIDER]
)

app.prepare(
    ctx_id=-1,
    det_size=DETECTION_SIZE
)

print("InsightFace Ready")


class FaceDetector:

    def __init__(self):
        self.model = app

    def detect(self, image):

        if image is None:
            return None

        img_h, img_w = image.shape[:2]

        faces = self.model.get(image)

        if len(faces) == 0:
            return None

        # Largest face
        face = max(
            faces,
            key=lambda f:
            (f.bbox[2] - f.bbox[0]) *
            (f.bbox[3] - f.bbox[1])
        )

        x1, y1, x2, y2 = map(int, face.bbox)

        face_width = x2 - x1
        face_height = y2 - y1
        face_area = face_width * face_height

        face_percent = (
            face_area /
            (img_w * img_h)
        ) * 100

        landmarks = face.kps.astype(np.int32)

        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        mouth_left = landmarks[3]
        mouth_right = landmarks[4]

        eye_distance = np.linalg.norm(
            right_eye - left_eye
        )

        mouth_distance = np.linalg.norm(
            mouth_right - mouth_left
        )

        eye_center = (
            int((left_eye[0] + right_eye[0]) / 2),
            int((left_eye[1] + right_eye[1]) / 2)
        )

        mouth_center = (
            int((mouth_left[0] + mouth_right[0]) / 2),
            int((mouth_left[1] + mouth_right[1]) / 2)
        )

        face_center = (
            int((x1 + x2) / 2),
            int((y1 + y2) / 2)
        )

        angle = math.degrees(
            math.atan2(
                right_eye[1] - left_eye[1],
                right_eye[0] - left_eye[0]
            )
        )

        # Blur Score
        gray = image[
            max(0, y1):min(img_h, y2),
            max(0, x1):min(img_w, x2)
        ]

        if gray.size != 0:

            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

            blur_score = cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()

            brightness = np.mean(gray)

        else:

            blur_score = 0
            brightness = 0

        return {

            # Face Count
            "face_count": len(faces),

            # Confidence
            "confidence": float(face.det_score),

            # Image
            "image_width": img_w,
            "image_height": img_h,

            # Bounding Box
            "bbox": (
                x1,
                y1,
                x2,
                y2
            ),

            # Face Size
            "width": face_width,
            "height": face_height,
            "area": face_area,
            "face_percent": round(face_percent, 2),

            # Centers
            "face_center": face_center,
            "eye_center": eye_center,
            "mouth_center": mouth_center,

            # Landmarks
            "left_eye": tuple(left_eye),
            "right_eye": tuple(right_eye),
            "nose": tuple(nose),
            "mouth_left": tuple(mouth_left),
            "mouth_right": tuple(mouth_right),
            "landmarks": landmarks,

            # Measurements
            "eye_distance": float(eye_distance),
            "mouth_distance": float(mouth_distance),

            # Rotation
            "angle": float(angle),

            # Quality
            "blur_score": round(float(blur_score), 2),
            "brightness": round(float(brightness), 2)
        }