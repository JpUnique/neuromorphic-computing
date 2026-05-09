import numpy as np
from snn_model.lif_neuron import LIFNeuron
from snn_model.stdp import STDP

class SimpleSNN:
    def __init__(self, num_neurons):
        # ✅ Create neurons with slight variation (break symmetry)
        self.neurons = [
            LIFNeuron(
                threshold=np.random.uniform(0.8, 1.2),
                decay=np.random.uniform(0.85, 0.95)
            )
            for _ in range(num_neurons)
        ]

        # ✅ Random initial weights
        self.weights = np.random.normal(loc=1.0, scale=0.1, size=num_neurons)

        # ✅ STDP learning rule
        self.stdp = STDP(lr=0.01)

    def forward(self, spikes):
        output = []

        for spike in spikes:
            neuron_outputs = []

            for i, neuron in enumerate(self.neurons):
                # ✅ Add small noise to break identical behavior
                noise = np.random.normal(0, 0.01)

                # ✅ FIXED INDENTATION (this was broken before)
                weighted_input = spike * self.weights[i] + noise

                # ✅ Neuron step
                out = neuron.step(weighted_input)

                # ✅ STDP learning update
                self.weights[i] = self.stdp.update(
                    self.weights[i], spike, out
                )

                neuron_outputs.append(out)

            output.append(neuron_outputs)

        return output
