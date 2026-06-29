import cv2
import math
import numpy as np


class CropEngine:

    def __init__(self,
                 output_width=300,
                 output_height=400):

        self.output_width = output_width
        self.output_height = output_height

        # Passport Output

        self.eye_position = 0.38

        self.face_ratio = 0.68

        self.aspect_ratio = output_width / output_height

        self.min_face_percent = 8

        self.max_face_percent = 35

    # =====================================================
    # Helpers
    # =====================================================

    def distance(self, p1, p2):

        return math.sqrt(

            (p1[0] - p2[0]) ** 2 +

            (p1[1] - p2[1]) ** 2

        )

    def midpoint(self, p1, p2):

        return (

            (p1[0] + p2[0]) / 2,

            (p1[1] + p2[1]) / 2

        )

    # =====================================================
    # Geometry
    # =====================================================

    def get_geometry(self, face):

        x1, y1, x2, y2 = face["bbox"]

        left_eye = face["left_eye"]

        right_eye = face["right_eye"]

        nose = face["nose"]

        mouth_left = face["mouth_left"]

        mouth_right = face["mouth_right"]

        face_width = x2 - x1

        face_height = y2 - y1

        eye_center = self.midpoint(

            left_eye,

            right_eye

        )

        mouth_center = self.midpoint(

            mouth_left,

            mouth_right

        )

        face_center = (

            (x1 + x2) / 2,

            (y1 + y2) / 2

        )

        eye_distance = self.distance(

            left_eye,

            right_eye

        )

        mouth_distance = self.distance(

            mouth_left,

            mouth_right

        )

        angle = math.degrees(

            math.atan2(

                right_eye[1] - left_eye[1],

                right_eye[0] - left_eye[0]

            )

        )

        return {

            "bbox": (x1, y1, x2, y2),

            "face_width": face_width,

            "face_height": face_height,

            "face_center": face_center,

            "eye_center": eye_center,

            "mouth_center": mouth_center,

            "eye_distance": eye_distance,

            "mouth_distance": mouth_distance,

            "angle": angle,

            "nose": nose

        }

    # =====================================================
    # Calculate Face Scale
    # =====================================================

    def estimate_face_width(self, geo):

        """
        Eye distance se actual head width estimate.

        Ye value dataset se tune hogi.
        """

        return geo["eye_distance"] * 2.20

    # =====================================================
    # Calculate Crop Size
    # =====================================================

    def calculate_crop_size(self, geo):

        estimated_face = self.estimate_face_width(geo)

        crop_width = estimated_face / self.face_ratio

        crop_height = crop_width * 4 / 3

        return crop_width, crop_height

    # =====================================================
    # Calculate Crop Rectangle
    # =====================================================

    def calculate_crop_rect(self, geo):

        crop_width, crop_height = self.calculate_crop_size(geo)

        eye_x, eye_y = geo["eye_center"]

        crop_x = eye_x - crop_width / 2

        crop_y = eye_y - crop_height * self.eye_position

        return (

            int(crop_x),

            int(crop_y),

            int(crop_x + crop_width),

            int(crop_y + crop_height)

        )
    
        # =====================================================
    # Add Padding
    # =====================================================

    def add_padding(self,
                    image,
                    left,
                    top,
                    right,
                    bottom):

        return cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_REPLICATE
        )

    # =====================================================
    # Safe Crop
    # =====================================================

    def safe_crop(self,
                  image,
                  rect):

        h, w = image.shape[:2]

        x1, y1, x2, y2 = rect

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w)
        pad_bottom = max(0, y2 - h)

        if pad_left or pad_top or pad_right or pad_bottom:

            image = self.add_padding(
                image,
                pad_left,
                pad_top,
                pad_right,
                pad_bottom
            )

            x1 += pad_left
            x2 += pad_left

            y1 += pad_top
            y2 += pad_top

        crop = image[
            y1:y2,
            x1:x2
        ]

        return crop

    # =====================================================
    # Remove White Border
    # =====================================================

    def remove_white_border(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        mask = gray < 245

        pts = np.argwhere(mask)

        if len(pts) == 0:
            return image

        y1, x1 = pts.min(axis=0)
        y2, x2 = pts.max(axis=0)

        return image[
            y1:y2,
            x1:x2
        ]

    # =====================================================
    # Normalize Size
    # =====================================================

    def normalize(self,
                  crop):

        crop = cv2.resize(

            crop,

            (

                self.output_width,

                self.output_height

            ),

            interpolation=cv2.INTER_CUBIC

        )

        return crop

    # =====================================================
    # Crop Image
    # =====================================================

    def crop_image(self,
                   image,
                   face):

        geo = self.get_geometry(face)

        rect = self.calculate_crop_rect(geo)

        crop = self.safe_crop(

            image,

            rect

        )

        crop = self.remove_white_border(crop)

        crop = self.normalize(crop)

        return crop
    
        # =====================================================
    # Smart Head Scale
    # =====================================================

    def smart_scale(self, crop):

        h, w = crop.shape[:2]

        # Safety
        if h == 0 or w == 0:
            return crop

        return crop

    # =====================================================
    # Final Crop API
    # =====================================================

    def crop(self,
             image,
             face):

        if image is None:
            return None

        if face is None:
            return None

        crop = self.crop_image(
            image,
            face
        )

        crop = self.smart_scale(
            crop
        )

        return crop

    # =====================================================
    # Debug Draw (Optional)
    # =====================================================

    def draw_debug(self,
                   image,
                   face):

        img = image.copy()

        x1, y1, x2, y2 = face["bbox"]

        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

        for key in [
            "left_eye",
            "right_eye",
            "nose",
            "mouth_left",
            "mouth_right"
        ]:

            x, y = face[key]

            cv2.circle(
                img,
                (int(x), int(y)),
                3,
                (0, 0, 255),
                -1
            )

        return img