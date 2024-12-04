import qiskit
import random
from ..backend import utilities, constant
from ..core import random_circuit

def specific_mutate(qc: qiskit.QuantumCircuit, pool, index: int) -> qiskit.QuantumCircuit:
    """Replace a quantum gate at specific index by another

    Args:
        - qc (qiskit.QuantumCircuit): input circuit
        - index (int): from 0 to num_gate - 1

    Returns:
        - qiskit.QuantumCircuit: Bit flipped circut
    """
    while (True):
        new_gate = random.choice(pool)
        if new_gate['num_params'] == 0:
            gate = new_gate['operation']()
        elif new_gate['num_params'] == 1:
            gate = new_gate['operation'](qiskit.circuit.Parameter(f'{index}'))
        else:
            gate = new_gate['operation'](qiskit.circuit.ParameterVector(f'{index}', new_gate['num_params']))
        if gate.num_qubits == qc.data[index].operation.num_qubits:
            break
    target_qubits = qc.data[index][1]
    if gate.num_qubits == 1:
        qc.data[index] = (gate, [target_qubits[0]], [])
    elif gate.num_qubits == 2:
        qc.data[index] = (gate, [target_qubits[0], target_qubits[1]] if len(target_qubits) > 1 else [target_qubits[0]], [])
    return qc

def bitflip_mutate(pool, prob_mutate: float = 0.1) -> qiskit.QuantumCircuit:
    """Mutate at every position in circuit with probability = prob_mutate

    Args:
        - qc (qiskit.QuantumCircuit): Input circuit
        - prob_mutate (float, optional): Mutate probability. Defaults to 0.1.

    Returns:
        - qiskit.QuantumCircuit: Bit flipped circuit
    """
    def bitflip_mutate_func(qc: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
        num_gates = len(qc.data)
        for index in range(0, num_gates):
            if random.random() < prob_mutate:
                qc = specific_mutate(qc, pool, index = index)  
        return qc
    return bitflip_mutate_func


def layerflip_mutate(qc: qiskit.QuantumCircuit, prob_mutate: float = 0.1) -> qiskit.QuantumCircuit:
    """Mutate qc to other.

    Args:
        qc (qiskit.QuantumCircuit)
        is_truncate (bool, optional): If it's true, make the qc depth into default. Defaults to True.

    Returns:
        qsee.evolution.eEqc: Mutatant
    """
    standard_depth = qc.depth()
    for index in range(0, standard_depth):
        if random.random() < prob_mutate:
            qc1, qc2 = utilities.divide_circuit_by_depth(qc, index)
            qc21, qc22 = utilities.divide_circuit_by_depth(qc2, 1)
            genome = random_circuit.generate_with_pool(qc.num_qubits, 1)
            qc = utilities.compose_circuit([qc1, genome, qc22])
            qc = utilities.truncate_circuit(qc.copy(), standard_depth)
    return qc