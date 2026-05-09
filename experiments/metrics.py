import numpy as np
import time


def measure_latency(encoder, snn, signal):
    start_time = time.time()

    spikes = encoder.encode(signal)
    _ = snn.forward(spikes)

    end_time = time.time()

    latency = (end_time - start_time) * 1000  # ms
    return latency


def measure_sparsity(spikes):
    total_spikes = np.sum(spikes)
    sparsity = 1 - (total_spikes / len(spikes))
    return sparsity


def measure_energy(spikes, energy_per_spike=0.001):
    """
    Simple energy model:
    energy = number_of_spikes × cost_per_spike
    """
    total_spikes = np.sum(spikes)
    energy = total_spikes * energy_per_spike
    return energy


def weight_change(initial_weights, final_weights):
    return final_weights - initial_weights
