import time

# Building a context manager is a more dinamic way to count time then to create a start time
# variable before and after the code we want to measure
class TimeMesure:
    def __enter__(self):
        self.start = time.perf_counter()  # high-precision timer
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.elapsed_ms = (self.end - self.start) * 1000  # convert to ms