import time
from collections import deque

class FPSCounter:
    def __init__(self, maxlen=60):
        # store timestamps of recent frames
        self.timestamps = deque(maxlen=maxlen)

    def update(self):
        """Call this once per frame"""
        now = time.time()
        self.timestamps.append(now)

    def get_fps(self):
        """Return the smoothed FPS"""
        if len(self.timestamps) < 2:
            return 0.0
        # time between first and last frame in queue
        duration = self.timestamps[-1] - self.timestamps[0]
        # frames divided by duration
        return (len(self.timestamps) - 1) / duration if duration > 0 else 0.0

fps_tracker = FPSCounter(30)