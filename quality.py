import cv2
import numpy as np


class QualityEngine:

    def __init__(self):

        self.min_face_percent = 8
        self.max_face_percent = 40

        self.min_blur = 80

        self.min_brightness = 45
        self.max_brightness = 220

        self.max_face_angle = 20

    def check(self, image, face):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        blur = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        brightness = np.mean(gray)

        reasons = []

        score = 100

        # -----------------------------
        # Face Size
        # -----------------------------

        if face["face_percent"] < self.min_face_percent:

            reasons.append("Face Too Small")

            score -= 20

        if face["face_percent"] > self.max_face_percent:

            reasons.append("Face Too Large")

            score -= 20

        # -----------------------------
        # Blur
        # -----------------------------

        if blur < self.min_blur:

            reasons.append("Blur Image")

            score -= 25

        # -----------------------------
        # Brightness
        # -----------------------------

        if brightness < self.min_brightness:

            reasons.append("Too Dark")

            score -= 15

        if brightness > self.max_brightness:

            reasons.append("Too Bright")

            score -= 15

        # -----------------------------
        # Rotation
        # -----------------------------

        if abs(face["angle"]) > self.max_face_angle:

            reasons.append("Large Face Rotation")

            score -= 20

        score = max(score, 0)

        return {

            "accepted": score >= 60,

            "score": score,

            "blur": round(blur, 2),

            "brightness": round(brightness, 2),

            "reasons": reasons

        }