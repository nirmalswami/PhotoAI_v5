import cv2
import numpy as np


def align_face(image, left_eye, right_eye):

    left_eye = np.array(left_eye, dtype=np.float32)
    right_eye = np.array(right_eye, dtype=np.float32)

    # Eyes center
    eyes_center = (
        (left_eye[0] + right_eye[0]) / 2,
        (left_eye[1] + right_eye[1]) / 2
    )

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    angle = np.degrees(np.arctan2(dy, dx))

    rotate_matrix = cv2.getRotationMatrix2D(
        eyes_center,
        angle,
        1.0
    )

    aligned = cv2.warpAffine(
        image,
        rotate_matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return aligned, rotate_matrix