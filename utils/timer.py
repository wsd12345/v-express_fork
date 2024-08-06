import time


class Timer:
    def __init__(self):
        self.time = 0
        self.st = 0
        self.et = 0

    def __enter__(self):
        self.st = time.monotonic()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.et = time.monotonic()
        self.time = self.et - self.st + self.time
