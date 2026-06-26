# ===============================
# PhotoAI v4 Configuration
# ===============================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "cropped")
REJECTED_DIR = os.path.join(BASE_DIR, "rejected")
LOG_DIR = os.path.join(BASE_DIR, "logs")

PROCESSED_FILE = os.path.join(BASE_DIR, "processed.txt")

OUTPUT_WIDTH = 300
OUTPUT_HEIGHT = 400

DETECTION_SIZE = (640, 640)

CPU_PROVIDER = "CPUExecutionProvider"