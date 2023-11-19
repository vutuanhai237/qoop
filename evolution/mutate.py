import qiskit
import random
from ..backend import utilities
from ..core import random_circuit
from .ecircuit import ECircuit

def bitflip_mutate(circuit: ECircuit, pool, is_truncate=True) -> ECircuit:
    """Mutate circuit to other.

    Args:
        circuit (qsee.evolution.ecircuit.ECircuit)
        pool (_type_): Selected gates
        is_truncate (bool, optional): If it's true, make the circuit depth into default. Defaults to True.

    Returns:
        qsee.evolution.ecircuit.ECircuit: Mutatant
    """
    standard_depth = circuit.qc.depth()
    qc1, qc2 = utilities.divide_circuit_by_depth(circuit.qc, random.randint(0, standard_depth - 1))
    qc21, qc22 = utilities.divide_circuit_by_depth(qc2, 1)
    genome = random_circuit.generate_with_pool(circuit.qc.num_qubits, 1, pool)
    new_qc = utilities.compose_circuit([qc1, genome, qc22])
    if is_truncate:
        new_qc = utilities.truncate_circuit(new_qc.copy(), standard_depth)
    circuit.qc = new_qc
    return circuit
