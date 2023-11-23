import types
import qiskit

class ECircuit():
    def __init__(self, qc: qiskit.QuantumCircuit, fitness_func: types.FunctionType) -> None:
        """Enhanced qiskit circuit with fitness properties

        Args:
            qc (qiskit.QuantumCircuit)
            fitness_func (types.FunctionType)
        """
        self.qc = qc
        self.fitness_func = fitness_func
        self.fitness = None
        self.true_fitness = None
        self.strength_point = 0
        return
    def compile(self):
        """Run fitness function to compute fitness value
        """
        if self.fitness is None:
            self.fitness = self.fitness_func(self.qc)
        return
    def true_compile(self):
        if self.true_fitness is None:
            self.true_fitness = self.fitness_func(self.qc, num_steps = 100)
        return