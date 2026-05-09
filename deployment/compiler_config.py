class HardwareConfig:
    def __init__(self):
        self.max_neurons_per_core = 256
        self.max_synapses_per_core = 10000
        self.communication_latency_us = 2  # microseconds

    def validate_mapping(self, mapping):
        """
        Check if mapping violates constraints
        """
        for core, neurons in mapping.items():
            if len(neurons) > self.max_neurons_per_core:
                print(f"Warning: Core {core} overloaded!")

        print("Mapping validation complete.")