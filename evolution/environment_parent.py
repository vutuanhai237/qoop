
from dataclasses import dataclass, field
from dataclasses import asdict

@dataclass
class Metadata:
    num_qubits: int
    num_circuit: int
    num_generation: int
    current_generation: int = field(default_factory=lambda: 0)
    fitnessss: list = field(default_factory=lambda: [])
    best_fitnesss: list = field(default_factory=lambda: [])
    prob_mutate: float = field(default_factory=lambda: 0.1)
    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}
