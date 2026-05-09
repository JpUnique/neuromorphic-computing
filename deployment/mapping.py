import numpy as np

class NeuromorphicMapper:
    def __init__(self, num_cores):
        self.num_cores = num_cores

    def map_neurons(self, num_neurons):
        """
        Simulate mapping neurons to cores
        """
        mapping = {}

        neurons_per_core = int(np.ceil(num_neurons / self.num_cores))

        neuron_id = 0
        for core in range(self.num_cores):
            assigned = []

            for _ in range(neurons_per_core):
                if neuron_id < num_neurons:
                    assigned.append(neuron_id)
                    neuron_id += 1

            mapping[core] = assigned

        return mapping