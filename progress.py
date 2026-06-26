import time

class Progress:

    def __init__(self, total):

        self.total = total
        self.done = 0
        self.start = time.time()

    def update(self):

        self.done += 1

        elapsed = time.time() - self.start

        speed = self.done / elapsed if elapsed else 0

        remain = self.total - self.done

        eta = remain / speed if speed else 0

        print(
            f"{self.done}/{self.total}"
            f" | {speed:.2f} img/sec"
            f" | ETA {eta/60:.1f} min"
        )