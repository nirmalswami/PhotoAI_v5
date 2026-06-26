import mediapipe as mp

class MeshDetector:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def detect(self, image):

        rgb = image[:, :, ::-1]

        result = self.mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        h, w = image.shape[:2]

        landmarks = []

        for lm in result.multi_face_landmarks[0].landmark:

            landmarks.append(
                (
                    int(lm.x * w),
                    int(lm.y * h)
                )
            )

        return landmarks