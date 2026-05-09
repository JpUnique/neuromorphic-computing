import numpy as np

class DeltaEncoder:
    def __init__(self, threshold):
        self.threshold = threshold
        self.prev_value = None

    def encode(self, signal):
        spikes = []

        for value in signal:
            if self.prev_value is None:
                self.prev_value = value
                spikes.append(0)
                continue

            if abs(value - self.prev_value) > self.threshold:
                spikes.append(1)
            else:
                spikes.append(0)

            self.prev_value = value

        return np.array(spikes)