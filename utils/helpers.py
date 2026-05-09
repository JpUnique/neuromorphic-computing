class LIFNeuron:
    def __init__(self, threshold=1.0, decay=0.9):
        self.threshold = threshold
        self.decay = decay
        self.membrane_potential = 0

    def step(self, input_spike):
        # Integrate
        self.membrane_potential += input_spike

        # Leak
        self.membrane_potential *= self.decay

        # Fire
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = 0
            return 1
        return 0