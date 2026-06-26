import os

def load_processed(file_path):

    if not os.path.exists(file_path):
        return set()

    with open(file_path, "r", encoding="utf-8") as f:
        return set(
            line.strip()
            for line in f
            if line.strip()
        )


def mark_processed(file_path, filename):

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(filename + "\n")