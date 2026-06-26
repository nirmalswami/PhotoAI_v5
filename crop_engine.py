import cv2
import math
import numpy as np


class CropEngine:

    def __init__(self,
             output_width=300,
             output_height=400):

    self.output_width = output_width
    self.output_height = output_height

    # Passport Standard
    self.eye_position = 0.38

    # Target face size
    self.target_face_ratio = 0.68

    # Safety margins
    self.top_margin = 0.22
    self.bottom_margin = 0.30

    # =====================================================
    # Geometry
    # =====================================================

    def get_geometry(self, face):

        x1, y1, x2, y2 = face["bbox"]

        left_eye = face["left_eye"]
        right_eye = face["right_eye"]

        face_width = x2 - x1
        face_height = y2 - y1

        face_center = (
            (x1 + x2) / 2,
            (y1 + y2) / 2
        )

        eye_center = (
            (left_eye[0] + right_eye[0]) / 2,
            (left_eye[1] + right_eye[1]) / 2
        )

        eye_distance = math.sqrt(
            (right_eye[0] - left_eye[0]) ** 2 +
            (right_eye[1] - left_eye[1]) ** 2
        )

        angle = math.degrees(

            math.atan2(

                right_eye[1] - left_eye[1],

                right_eye[0] - left_eye[0]

            )
            

        )

        return {

            "face_width": face_width,

            "face_height": face_height,

            "face_center": face_center,

            "eye_center": eye_center,

            "eye_distance": eye_distance,

            "angle": angle

        }

    # =====================================================
    # Rotate Image
    # =====================================================

    def rotate_image(self,
                     image,
                     center,
                     angle):

        h, w = image.shape[:2]

        M = cv2.getRotationMatrix2D(

            center,

            angle,

            1.0

        )

        rotated = cv2.warpAffine(

            image,

            M,

            (w, h),

            flags=cv2.INTER_CUBIC,

            borderMode=cv2.BORDER_REPLICATE

        )

        return rotated, M

    # =====================================================
    # Transform Point
    # =====================================================

    def transform_point(self,
                        point,
                        matrix):

        x = point[0]
        y = point[1]

        new_x = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
        new_y = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]

        return (new_x, new_y)

    # =====================================================
    # Rotate all landmarks
    # =====================================================

    def rotate_landmarks(self,
                         face,
                         matrix):

        result = {}

        result["bbox"] = face["bbox"]

        result["left_eye"] = self.transform_point(
            face["left_eye"],
            matrix
        )

        result["right_eye"] = self.transform_point(
            face["right_eye"],
            matrix
        )

        result["nose"] = self.transform_point(
            face["nose"],
            matrix
        )

        result["mouth_left"] = self.transform_point(
            face["mouth_left"],
            matrix
        )

        result["mouth_right"] = self.transform_point(
            face["mouth_right"],
            matrix
        )

        if "kps" in face:

            pts = []

            for p in face["kps"]:

                pts.append(

                    self.transform_point(
                        p,
                        matrix
                    )

                )

            result["kps"] = np.array(pts)

        return result

    # =====================================================
    # Align Face
    # =====================================================

    def align_face(self,
                   image,
                   face):

        geo = self.get_geometry(face)

        rotated, matrix = self.rotate_image(

            image,

            geo["eye_center"],

            geo["angle"]

        )

        new_face = self.rotate_landmarks(

            face,

            matrix

        )

        return rotated, new_face

    # =====================================================
    # Crop (Placeholder)
    # =====================================================

    def crop(self,
             image,
             face):

        rotated_image, rotated_face = self.align_face(

            image,

            face

        )

        return rotated_image