import numpy as np
import qiskit
import random
from ..backend import constant
from .environment_parent import Metadata
def initialize_random_parameters(num_qubits: int, max_operands: int, conditional: bool, seed):
    if max_operands < 1 or max_operands > 3:
        raise qiskit.circuit.exceptions.CircuitError("max_operands must be between 1 and 3")

    qr = qiskit.circuit.QuantumRegister(num_qubits, 'q')
    qc = qiskit.circuit.QuantumCircuit(num_qubits)

    if conditional:
        cr = qiskit.circuit.ClassicalRegister(num_qubits, 'c')
        qc.add_register(cr)

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)

    rng = np.random.default_rng(seed)
    thetas = qiskit.circuit.ParameterVector('theta')
    return qr, qc, rng, thetas
    
    
def choice_from_array(arr, condition):
    item = None
    while item is None:
        item = random.choice(arr)
        if condition(item):
            return item
        else:
            item = None
    return item

def by_depth(metadata: Metadata) -> qiskit.QuantumCircuit:
    num_qubits = metadata.num_qubits
    depth = metadata.depth
    pool = constant.operations
    conditional = False
    seed=None
    max_operands = 2
    qr, qc, rng, thetas = initialize_random_parameters(num_qubits, max_operands, conditional, seed)
    thetas_length = 0
    for _ in range(depth):
        remaining_qubits = list(range(num_qubits))
        while remaining_qubits:
            max_possible_operands = min(len(remaining_qubits), max_operands)
            num_operands = choice_from_array(
                [1, 1, 1, 1, 1, 1, 2, 2, 2, 2], lambda value: value <= max_possible_operands)
            rng.shuffle(remaining_qubits)
            operands = remaining_qubits[:num_operands]
            remaining_qubits = [
                q for q in remaining_qubits if q not in operands]
            num_op_pool = [
                item for item in pool if item['num_op'] == num_operands]

            operation = rng.choice(num_op_pool)
            num_params = operation['num_params']
            thetas_length += num_params
            thetas.resize(thetas_length)
            angles = thetas[thetas_length - num_params:thetas_length]
            register_operands = [qr[i] for i in operands]
            op = operation['operation'](*angles)
            qc.append(op, register_operands)
    return qc

def weighted_choice(arr, weights):
    if len(arr) != len(weights):
        raise ValueError("The length of the arrays must be the same")
    if sum(weights) != 1:
        raise ValueError("The sum of the weights must be 1")
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    return np.random.choice(arr, p=normalized_weights)



def by_num_cnot(metadata: Metadata) -> qiskit.QuantumCircuit:
    num_qubits = metadata.num_qubits
    depth = metadata.depth
    pool = constant.operations_only_cnot
    conditional = False
    seed=None
    max_operands = 2
    qr, qc, rng, thetas = initialize_random_parameters(num_qubits, max_operands, conditional, seed)
    thetas_length = 0
    num_current_cnot = 0
    percent_cnot = 0.1 + 0.1 * np.random.randint(0, 3)
    for _ in range(depth):
        remaining_qubits = list(range(num_qubits))
        while remaining_qubits:
            max_possible_operands = min(len(remaining_qubits), max_operands)
            if num_current_cnot == metadata.num_cnot:
                max_possible_operands = 1
            else:
                if max_possible_operands == 2:
                    num_current_cnot += 1
            if max_possible_operands == 1:
                num_operands = 1
            else:
                num_operands = weighted_choice([1, 2], [1 - percent_cnot, percent_cnot])
            rng.shuffle(remaining_qubits)
            operands = remaining_qubits[:num_operands]
            remaining_qubits = [
                q for q in remaining_qubits if q not in operands]
            num_op_pool = [
                item for item in pool if item['num_op'] == num_operands]

            operation = rng.choice(num_op_pool)
            num_params = operation['num_params']
            thetas_length += num_params
            thetas.resize(thetas_length)
            angles = thetas[thetas_length - num_params:thetas_length]
            register_operands = [qr[i] for i in operands]
            op = operation['operation'](*angles)
            qc.append(op, register_operands)
    return qc