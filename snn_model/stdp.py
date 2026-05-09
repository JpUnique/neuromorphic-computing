import numpy as np

class STDP:
    def __init__(self, lr=0.01):
        self.lr = lr  # learning rate

    def update(self, weights, pre_spike, post_spike):
        """
        Simple STDP rule:
        - If pre fires before post → increase weight
        - If post fires without pre → decrease weight
        """
        if pre_spike == 1 and post_spike == 1:
            weights += self.lr   # potentiation
        elif pre_spike == 0 and post_spike == 1:
            weights -= self.lr   # depression

        return weights