import numpy as np
import matplotlib.pyplot as plt

from encoding.delta_encoder import DeltaEncoder
from snn_model.network import SimpleSNN
from experiments.metrics import (
    measure_latency,
    measure_sparsity,
    measure_energy,
    weight_change
)
from deployment.mapping import NeuromorphicMapper
from deployment.compiler_config import HardwareConfig

# -----------------------------
# 1. Generate Sample Signal
# -----------------------------
t = np.linspace(0, 10, 100)
rng = np.random.default_rng(0)
signal = np.sin(t) + 0.1 * rng.standard_normal(100)

# -----------------------------
# 2. Encoding (Delta Modulation)
# -----------------------------
encoder = DeltaEncoder(threshold=0.2)
spikes = encoder.encode(signal)

# -----------------------------
# 3. Initialize SNN
# -----------------------------
snn = SimpleSNN(num_neurons=5)

# ✅ Store initial weights BEFORE learning
initial_weights = snn.weights.copy()

# -----------------------------
# 4. Run SNN
# -----------------------------
output = snn.forward(spikes)
output_array = np.array(output)

# -----------------------------
# 5. Metrics Calculation
# -----------------------------
total_spikes = np.sum(spikes)
sparsity = measure_sparsity(spikes)
energy = measure_energy(spikes)
latency = measure_latency(encoder, snn, signal)

total_firings = np.sum(output_array)
neuron_activity = np.sum(output_array, axis=0)

# ✅ Learning effect
weight_diff = weight_change(initial_weights, snn.weights)

# -----------------------------
# 6. Print Results
# -----------------------------
print("\n--- BASIC STATS ---")
print("Total spikes:", total_spikes)
print("Sparsity:", sparsity)

print("\n--- NEURAL ACTIVITY ---")
print("Total neuron firings:", total_firings)
print("Neuron activity per neuron:", neuron_activity)

print("\n--- LEARNING ---")
print("Final synaptic weights:", snn.weights)
print("Weight changes:", weight_diff)

print("\n--- PERFORMANCE METRICS ---")
print("Latency (ms):", latency)
print("Energy estimate:", energy)

# -----------------------------
# 7. Visualization - Signal & Spikes
# -----------------------------
plt.figure(figsize=(12, 5))

plt.subplot(2, 1, 1)
plt.plot(signal)
plt.title("Original Signal")

plt.subplot(2, 1, 2)
plt.step(range(len(spikes)), spikes)
plt.title("Spike Encoding")

plt.tight_layout()
plt.show()

# -----------------------------
# 8. Neuron Heatmap
# -----------------------------
plt.figure(figsize=(10, 4))

plt.imshow(output_array.T, cmap='gray', aspect='auto')
plt.title("Neuron Firing Activity Heatmap")
plt.xlabel("Time Step")
plt.ylabel("Neuron Index")
plt.colorbar(label="Spike")

plt.show()

# -----------------------------
# 9. Final Weights Bar Chart
# -----------------------------
plt.figure()

plt.bar(range(len(snn.weights)), snn.weights)
plt.title("Final Synaptic Weights")
plt.xlabel("Neuron Index")
plt.ylabel("Weight Value")

plt.show()

# -----------------------------
# 10. Weight Change Plot ✅
# -----------------------------
plt.figure()

plt.bar(range(len(weight_diff)), weight_diff)
plt.title("Synaptic Weight Changes (STDP Effect)")
plt.xlabel("Neuron Index")
plt.ylabel("Weight Change")

plt.show()

# -----------------------------
# 11. Performance Metrics Plot ✅
# -----------------------------
metrics_names = ["Latency (ms)", "Energy", "Sparsity"]
metrics_values = [latency, energy, sparsity]

plt.figure()

plt.bar(metrics_names, metrics_values)
plt.title("System Performance Metrics")
plt.ylabel("Value")

plt.show()

# -----------------------------
# 3.5 Hardware Mapping Simulation ✅
# -----------------------------
mapper = NeuromorphicMapper(num_cores=4)
mapping = mapper.map_neurons(num_neurons=5)

config = HardwareConfig()
config.validate_mapping(mapping)

print("\n--- HARDWARE MAPPING ---")
for core, neurons in mapping.items():
    print(f"Core {core}: Neurons {neurons}")