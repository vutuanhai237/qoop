
import logging
import qiskit
import numpy as np
import enum
from qiskit.circuit.library.standard_gates import (IGate, U1Gate, U2Gate, U3Gate, XGate,
                                                   YGate, ZGate, HGate, SGate, SdgGate, TGate,
                                                   TdgGate, RXGate, RYGate, RZGate, CXGate,
                                                   CYGate, CZGate, CHGate, CRXGate, CRYGate, CRZGate, CU1Gate,
                                                   CU3Gate, SwapGate, RZZGate,
                                                   CCXGate, CSwapGate)


# GA process
class GAMode(enum.Enum):
    NORMAL_MODE = 'normal_mode'
    PREDICTOR_MODE = 'predict_node'

class MeasureMode(enum.Enum):
    THEORY = 'theory'
    SIMULATE = 'simulate'
    
# Predictor hyperparameter
DROP_OUT_RATE = 0.5
L2_REGULARIZER_RATE = 0.005
NUM_EPOCH = 100
# Training hyperparameter
NUM_SHOTS = 10000
LEARNING_RATE = 0.1
NOISE_PROB = 0.0  # [0, 1]
GAMMA = 0.7  # learning rate decay rate
DELTA = 0.01  # minimum change value of loss value
DISCOUNTING_FACTOR = 0.3  # [0, 1]
# backend = qiskit.Aer.get_backend('statevector_simulator') 

# Logger

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


# For parameter-shift rule
two_term_psr = {
    'r': 1/2,
    's': np.pi / 2
}

four_term_psr = {
    'alpha': np.pi / 2,
    'beta': 3 * np.pi / 2,
    'd_plus': (np.sqrt(2) + 1) / (4*np.sqrt(2)),
    'd_minus': (np.sqrt(2) - 1) / (4*np.sqrt(2))
}

one_qubit_gates = ["Hadamard", 'RX', 'RY', 'RZ']
two_qubits_gates = ['CNOT', 'CY', 'CZ', 'CRX', 'CRY', 'CRZ']


def create_gate_pool(num_qubits: int, one_qubit_gates: typing.List[str] = one_qubit_gates, two_qubits_gates: typing.List[str] = two_qubits_gates) -> typing.List[object]:
    """Create a pool gate from list of 1-qubit and 2-qubits gate

    Args:
        num_qubits (int): Number of qubit
        one_qubit_gates (typing.List[str], optional): List of 1-qubit gate. Defaults to one_qubit_gates.
        two_qubits_gates (typing.List[str], optional): List of 2-qubits gate. Defaults to two_qubits_gates.

    Returns:
        typing.List[object]: Pool gate
    """
    gate_pool = []

    # Single-qubit gates
    single_qubit_gates = one_qubit_gates
    for qubit in range(num_qubits):
        for gate in single_qubit_gates:
            gate_pool.append((gate, qubit))

    # Two-qubit gates
    two_qubit_gates = two_qubits_gates
    for qubit1 in range(num_qubits):
        for qubit2 in range(num_qubits):
            if qubit1 != qubit2:
                for gate in two_qubit_gates:
                    gate_pool.append((gate, qubit1, qubit2))

    return gate_pool


# For QNG
generator = {
    'cu': -1 / 2 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'rx': -1 / 2 * np.array([[0, 1], [1, 0]], dtype=np.complex128),
    'crx': -1 / 2 * np.array([[0, 1], [1, 0]], dtype=np.complex128),
    'ry': -1 / 2 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'cry': -1 / 2 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'rz': -1 / 2 * np.array([[1, 0], [0, -1]], dtype=np.complex128),
    'crz': -1 / 2 * np.array([[1, 0], [0, -1]], dtype=np.complex128),
    'cz': -1 / 2 * np.array([[1, 0], [0, -1]], dtype=np.complex128),
    'i': np.array([[1, 0], [0, 1]], dtype=np.complex128),
    'id': np.array([[1, 0], [0, 1]], dtype=np.complex128),
    '11': np.array([[0, 0], [0, 1]], dtype=np.complex128),
}

ignore_generator = [
    'barrier'
]
parameterized_generator = [
    'rx', 'ry', 'rz', 'crx', 'cry', 'crz'
]

# This information is extracted from http://dx.doi.org/10.1103/PhysRevA.83.042314
edges_graph_state = {
    2: ["0-1"],
    3: ["0-1", "0-2"],
    4: ["0-2", "1-3", "2-3"],
    5: ["0-3", "2-4", "1-4", "2-3"],
    6: ["0-4", "1-5", "2-3", "2-4", "3-5"],
    7: ["0-5", "2-3", "4-6", "1-6", "2-4", "3-5"],
    8: ["0-7", "1-6", "2-4", "3-5", "2-3", "4-6", "5-7"],
    9: ["0-8", "2-3", "4-6", "5-7", "1-7", "2-4", "3-5", "6-8"],
    10: ["0-9", "1-8", "2-3", "4-6", "5-7", "2-4", "3-5", "6-9", "7-8"]
}

look_up_operator = {
    "Identity": 'I',
    "Hadamard": 'H',
    "PauliX": 'X',
    'PauliY': 'Y',
    'PauliZ': 'Z',
    'S': 'S',
    'T': 'T',
    'SX': 'SX',
    'CNOT': 'CX',
    'CZ': 'CZ',
    'CY': 'CY',
    'SWAP': 'SWAP',
    'ISWAP': 'ISWAP',
    'CSWAP': 'CSWAP',
    'Toffoli': 'CCX',
    'RX': 'RX',
    'RY': 'RY',
    'RZ': 'RZ',
    'CRX': 'CRX',
    'CRY': 'CRY',
    'CRZ': 'CRZ',
    'U1': 'U1',
    'U2': 'U2',
    'U3': 'U3',
    'IsingXX': 'RXX',
    'IsingYY': 'RYY',
    'IsingZZ': 'RZZ',
}

I_gate = {'name': 'i', 'operation': IGate, 'num_op': 1, 'num_params': 0}
H_gate = {'name': 'h', 'operation': HGate, 'num_op': 1, 'num_params': 0}
S_gate = {'name': 's', 'operation': SGate, 'num_op': 1, 'num_params': 0}
X_gate = {'name': 'x', 'operation': XGate, 'num_op': 1, 'num_params': 0}
Y_gate = {'name': 'y', 'operation': YGate, 'num_op': 1, 'num_params': 0}
Z_gate = {'name': 'z', 'operation': ZGate, 'num_op': 1, 'num_params': 0}
CX_gate = {'name': 'cx', 'operation': CXGate, 'num_op': 2, 'num_params': 0}
CRX_gate = {'name': 'crx', 'operation': CRXGate, 'num_op': 2, 'num_params': 1}
CRY_gate = {'name': 'cry', 'operation': CRYGate, 'num_op': 2, 'num_params': 1}
CRZ_gate = {'name': 'crz', 'operation': CRZGate, 'num_op': 2, 'num_params': 1}
RX_gate = {'name': 'rx', 'operation': RXGate, 'num_op': 1, 'num_params': 1}
RY_gate = {'name': 'ry', 'operation': RYGate, 'num_op': 1, 'num_params': 1}
RZ_gate = {'name': 'rz', 'operation': RZGate, 'num_op': 1, 'num_params': 1}
U2_gate = {'name': 'u2', 'operation': U2Gate, 'num_op': 1, 'num_params': 2}
U3_gate = {'name': 'u3', 'operation': U3Gate, 'num_op': 1, 'num_params': 3}

clifford_set = [
    H_gate,
    CX_gate,
    S_gate

]


full_operations = [
    I_gate,
    H_gate,
    S_gate,
    X_gate,
    Y_gate,
    Z_gate,
    CX_gate,
    RX_gate,
    RY_gate,
    RZ_gate,
    CRX_gate,
    CRY_gate,
    CRZ_gate
]

operations = [
    H_gate,
    # S_gate,
    # X_gate,
    # Y_gate,
    # Z_gate,
    CX_gate,
    RX_gate,
    RY_gate,
    RZ_gate,
    CRX_gate,
    CRY_gate,
    CRZ_gate
]

gate_distaces = [{'name_gate1': 'i', 'name_gate2': 'i', 'distance': 0},
                 {'name_gate1': 'i', 'name_gate2': 'h', 'distance': 2.0},
                 {'name_gate1': 'i', 'name_gate2': 's', 'distance': 1.4142},
                 {'name_gate1': 'i', 'name_gate2': 'x', 'distance': 2.0},
                 {'name_gate1': 'i', 'name_gate2': 'y', 'distance': 2.0},
                 {'name_gate1': 'i', 'name_gate2': 'z', 'distance': 2.0},
                 {'name_gate1': 'i', 'name_gate2': 'cx', 'distance': 2.0},
                 {'name_gate1': 'i', 'name_gate2': 'rx', 'distance': 1.7994},
                 {'name_gate1': 'i', 'name_gate2': 'ry', 'distance': 1.796},
                 {'name_gate1': 'i', 'name_gate2': 'rz', 'distance': 1.7985},
                 {'name_gate1': 'i', 'name_gate2': 'crx', 'distance': 1.798},
                 {'name_gate1': 'i', 'name_gate2': 'cry', 'distance': 1.7981},
                 {'name_gate1': 'i', 'name_gate2': 'crz', 'distance': 1.803},
                 {'name_gate1': 'h', 'name_gate2': 'i', 'distance': 2.0},
                 {'name_gate1': 'h', 'name_gate2': 'h', 'distance': 0},
                 {'name_gate1': 'h', 'name_gate2': 's', 'distance': 1.608},
                 {'name_gate1': 'h', 'name_gate2': 'x', 'distance': 1.0824},
                 {'name_gate1': 'h', 'name_gate2': 'y', 'distance': 2.0},
                 {'name_gate1': 'h', 'name_gate2': 'z', 'distance': 1.0824},
                 {'name_gate1': 'h', 'name_gate2': 'cx', 'distance': 2.2741},
                 {'name_gate1': 'h', 'name_gate2': 'rx', 'distance': 2.0},
                 {'name_gate1': 'h', 'name_gate2': 'ry', 'distance': 2.0},
                 {'name_gate1': 'h', 'name_gate2': 'rz', 'distance': 2.0},
                 {'name_gate1': 'h', 'name_gate2': 'crx', 'distance': 2.8284},
                 {'name_gate1': 'h', 'name_gate2': 'cry', 'distance': 2.8284},
                 {'name_gate1': 'h', 'name_gate2': 'crz', 'distance': 2.8284},
                 {'name_gate1': 's', 'name_gate2': 'i', 'distance': 1.4142},
                 {'name_gate1': 's', 'name_gate2': 'h', 'distance': 1.608},
                 {'name_gate1': 's', 'name_gate2': 's', 'distance': 0},
                 {'name_gate1': 's', 'name_gate2': 'x', 'distance': 2.0},
                 {'name_gate1': 's', 'name_gate2': 'y', 'distance': 2.0},
                 {'name_gate1': 's', 'name_gate2': 'z', 'distance': 1.4142},
                 {'name_gate1': 's', 'name_gate2': 'cx', 'distance': 2.4495},
                 {'name_gate1': 's', 'name_gate2': 'rx', 'distance': 1.9658},
                 {'name_gate1': 's', 'name_gate2': 'ry', 'distance': 1.9668},
                 {'name_gate1': 's', 'name_gate2': 'rz', 'distance': 1.59},
                 {'name_gate1': 's', 'name_gate2': 'crx', 'distance': 2.4319},
                 {'name_gate1': 's', 'name_gate2': 'cry', 'distance': 2.4325},
                 {'name_gate1': 's', 'name_gate2': 'crz', 'distance': 2.1489},
                 {'name_gate1': 'x', 'name_gate2': 'i', 'distance': 2.0},
                 {'name_gate1': 'x', 'name_gate2': 'h', 'distance': 1.0824},
                 {'name_gate1': 'x', 'name_gate2': 's', 'distance': 2.0},
                 {'name_gate1': 'x', 'name_gate2': 'x', 'distance': 0},
                 {'name_gate1': 'x', 'name_gate2': 'y', 'distance': 2.0},
                 {'name_gate1': 'x', 'name_gate2': 'z', 'distance': 2.0},
                 {'name_gate1': 'x', 'name_gate2': 'cx', 'distance': 2.0},
                 {'name_gate1': 'x', 'name_gate2': 'rx', 'distance': 2.0},
                 {'name_gate1': 'x', 'name_gate2': 'ry', 'distance': 2.0},
                 {'name_gate1': 'x', 'name_gate2': 'rz', 'distance': 2.0},
                 {'name_gate1': 'x', 'name_gate2': 'crx', 'distance': 2.8284},
                 {'name_gate1': 'x', 'name_gate2': 'cry', 'distance': 2.8284},
                 {'name_gate1': 'x', 'name_gate2': 'crz', 'distance': 2.8284},
                 {'name_gate1': 'y', 'name_gate2': 'i', 'distance': 2.0},
                 {'name_gate1': 'y', 'name_gate2': 'h', 'distance': 2.0},
                 {'name_gate1': 'y', 'name_gate2': 's', 'distance': 2.0},
                 {'name_gate1': 'y', 'name_gate2': 'x', 'distance': 2.0},
                 {'name_gate1': 'y', 'name_gate2': 'y', 'distance': 0},
                 {'name_gate1': 'y', 'name_gate2': 'z', 'distance': 2.0},
                 {'name_gate1': 'y', 'name_gate2': 'cx', 'distance': 2.8284},
                 {'name_gate1': 'y', 'name_gate2': 'rx', 'distance': 2.0},
                 {'name_gate1': 'y', 'name_gate2': 'ry', 'distance': 2.0},
                 {'name_gate1': 'y', 'name_gate2': 'rz', 'distance': 2.0},
                 {'name_gate1': 'y', 'name_gate2': 'crx', 'distance': 2.8284},
                 {'name_gate1': 'y', 'name_gate2': 'cry', 'distance': 2.8284},
                 {'name_gate1': 'y', 'name_gate2': 'crz', 'distance': 2.8284},
                 {'name_gate1': 'z', 'name_gate2': 'i', 'distance': 2.0},
                 {'name_gate1': 'z', 'name_gate2': 'h', 'distance': 1.0824},
                 {'name_gate1': 'z', 'name_gate2': 's', 'distance': 1.4142},
                 {'name_gate1': 'z', 'name_gate2': 'x', 'distance': 2.0},
                 {'name_gate1': 'z', 'name_gate2': 'y', 'distance': 2.0},
                 {'name_gate1': 'z', 'name_gate2': 'z', 'distance': 0},
                 {'name_gate1': 'z', 'name_gate2': 'cx', 'distance': 2.8284},
                 {'name_gate1': 'z', 'name_gate2': 'rx', 'distance': 2.0},
                 {'name_gate1': 'z', 'name_gate2': 'ry', 'distance': 2.0},
                 {'name_gate1': 'z', 'name_gate2': 'rz', 'distance': 2.0},
                 {'name_gate1': 'z', 'name_gate2': 'crx', 'distance': 2.8284},
                 {'name_gate1': 'z', 'name_gate2': 'cry', 'distance': 2.8284},
                 {'name_gate1': 'z', 'name_gate2': 'crz', 'distance': 2.8284},
                 {'name_gate1': 'cx', 'name_gate2': 'i', 'distance': 2.0},
                 {'name_gate1': 'cx', 'name_gate2': 'h', 'distance': 2.2741},
                 {'name_gate1': 'cx', 'name_gate2': 's', 'distance': 2.4495},
                 {'name_gate1': 'cx', 'name_gate2': 'x', 'distance': 2.0},
                 {'name_gate1': 'cx', 'name_gate2': 'y', 'distance': 2.8284},
                 {'name_gate1': 'cx', 'name_gate2': 'z', 'distance': 2.8284},
                 {'name_gate1': 'cx', 'name_gate2': 'cx', 'distance': 0},
                 {'name_gate1': 'cx', 'name_gate2': 'rx', 'distance': 2.7816},
                 {'name_gate1': 'cx', 'name_gate2': 'ry', 'distance': 2.7815},
                 {'name_gate1': 'cx', 'name_gate2': 'rz', 'distance': 2.783},
                 {'name_gate1': 'cx', 'name_gate2': 'crx', 'distance': 2.0},
                 {'name_gate1': 'cx', 'name_gate2': 'cry', 'distance': 2.0},
                 {'name_gate1': 'cx', 'name_gate2': 'crz', 'distance': 2.0},
                 {'name_gate1': 'rx', 'name_gate2': 'i', 'distance': 1.7954},
                 {'name_gate1': 'rx', 'name_gate2': 'h', 'distance': 2.0},
                 {'name_gate1': 'rx', 'name_gate2': 's', 'distance': 1.9666},
                 {'name_gate1': 'rx', 'name_gate2': 'x', 'distance': 2.0},
                 {'name_gate1': 'rx', 'name_gate2': 'y', 'distance': 2.0},
                 {'name_gate1': 'rx', 'name_gate2': 'z', 'distance': 2.0},
                 {'name_gate1': 'rx', 'name_gate2': 'cx', 'distance': 2.7819},
                 {'name_gate1': 'rx', 'name_gate2': 'rx', 'distance': 0},
                 {'name_gate1': 'rx', 'name_gate2': 'ry', 'distance': 1.2747},
                 {'name_gate1': 'rx', 'name_gate2': 'rz', 'distance': 1.2709},
                 {'name_gate1': 'rx', 'name_gate2': 'crx', 'distance': 1.8001},
                 {'name_gate1': 'rx', 'name_gate2': 'cry', 'distance': 2.2903},
                 {'name_gate1': 'rx', 'name_gate2': 'crz', 'distance': 2.2953},
                 {'name_gate1': 'ry', 'name_gate2': 'i', 'distance': 1.7951},
                 {'name_gate1': 'ry', 'name_gate2': 'h', 'distance': 2.0},
                 {'name_gate1': 'ry', 'name_gate2': 's', 'distance': 1.9667},
                 {'name_gate1': 'ry', 'name_gate2': 'x', 'distance': 2.0},
                 {'name_gate1': 'ry', 'name_gate2': 'y', 'distance': 2.0},
                 {'name_gate1': 'ry', 'name_gate2': 'z', 'distance': 2.0},
                 {'name_gate1': 'ry', 'name_gate2': 'cx', 'distance': 2.7818},
                 {'name_gate1': 'ry', 'name_gate2': 'rx', 'distance': 1.2738},
                 {'name_gate1': 'ry', 'name_gate2': 'ry', 'distance': 0},
                 {'name_gate1': 'ry', 'name_gate2': 'rz', 'distance': 1.2746},
                 {'name_gate1': 'ry', 'name_gate2': 'crx', 'distance': 2.2884},
                 {'name_gate1': 'ry', 'name_gate2': 'cry', 'distance': 1.8026},
                 {'name_gate1': 'ry', 'name_gate2': 'crz', 'distance': 2.2867},
                 {'name_gate1': 'rz', 'name_gate2': 'i', 'distance': 1.8019},
                 {'name_gate1': 'rz', 'name_gate2': 'h', 'distance': 2.0},
                 {'name_gate1': 'rz', 'name_gate2': 's', 'distance': 1.59},
                 {'name_gate1': 'rz', 'name_gate2': 'x', 'distance': 2.0},
                 {'name_gate1': 'rz', 'name_gate2': 'y', 'distance': 2.0},
                 {'name_gate1': 'rz', 'name_gate2': 'z', 'distance': 2.0},
                 {'name_gate1': 'rz', 'name_gate2': 'cx', 'distance': 2.7824},
                 {'name_gate1': 'rz', 'name_gate2': 'rx', 'distance': 1.2743},
                 {'name_gate1': 'rz', 'name_gate2': 'ry', 'distance': 1.2726},
                 {'name_gate1': 'rz', 'name_gate2': 'rz', 'distance': 0},
                 {'name_gate1': 'rz', 'name_gate2': 'crx', 'distance': 2.2878},
                 {'name_gate1': 'rz', 'name_gate2': 'cry', 'distance': 2.2873},
                 {'name_gate1': 'rz', 'name_gate2': 'crz', 'distance': 1.7992},
                 {'name_gate1': 'crx', 'name_gate2': 'i', 'distance': 1.7995},
                 {'name_gate1': 'crx', 'name_gate2': 'h', 'distance': 2.8284},
                 {'name_gate1': 'crx', 'name_gate2': 's', 'distance': 2.4316},
                 {'name_gate1': 'crx', 'name_gate2': 'x', 'distance': 2.8284},
                 {'name_gate1': 'crx', 'name_gate2': 'y', 'distance': 2.8284},
                 {'name_gate1': 'crx', 'name_gate2': 'z', 'distance': 2.8284},
                 {'name_gate1': 'crx', 'name_gate2': 'cx', 'distance': 2.0},
                 {'name_gate1': 'crx', 'name_gate2': 'rx', 'distance': 1.8017},
                 {'name_gate1': 'crx', 'name_gate2': 'ry', 'distance': 2.2916},
                 {'name_gate1': 'crx', 'name_gate2': 'rz', 'distance': 2.2942},
                 {'name_gate1': 'crx', 'name_gate2': 'crx', 'distance': 0},
                 {'name_gate1': 'crx', 'name_gate2': 'cry', 'distance': 1.2742},
                 {'name_gate1': 'crx', 'name_gate2': 'crz', 'distance': 1.2731},
                 {'name_gate1': 'cry', 'name_gate2': 'i', 'distance': 1.7982},
                 {'name_gate1': 'cry', 'name_gate2': 'h', 'distance': 2.8284},
                 {'name_gate1': 'cry', 'name_gate2': 's', 'distance': 2.4316},
                 {'name_gate1': 'cry', 'name_gate2': 'x', 'distance': 2.8284},
                 {'name_gate1': 'cry', 'name_gate2': 'y', 'distance': 2.8284},
                 {'name_gate1': 'cry', 'name_gate2': 'z', 'distance': 2.8284},
                 {'name_gate1': 'cry', 'name_gate2': 'cx', 'distance': 2.0},
                 {'name_gate1': 'cry', 'name_gate2': 'rx', 'distance': 2.2872},
                 {'name_gate1': 'cry', 'name_gate2': 'ry', 'distance': 1.802},
                 {'name_gate1': 'cry', 'name_gate2': 'rz', 'distance': 2.2889},
                 {'name_gate1': 'cry', 'name_gate2': 'crx', 'distance': 1.2747},
                 {'name_gate1': 'cry', 'name_gate2': 'cry', 'distance': 0},
                 {'name_gate1': 'cry', 'name_gate2': 'crz', 'distance': 1.2697},
                 {'name_gate1': 'crz', 'name_gate2': 'i', 'distance': 1.8019},
                 {'name_gate1': 'crz', 'name_gate2': 'h', 'distance': 2.8284},
                 {'name_gate1': 'crz', 'name_gate2': 's', 'distance': 2.1478},
                 {'name_gate1': 'crz', 'name_gate2': 'x', 'distance': 2.8284},
                 {'name_gate1': 'crz', 'name_gate2': 'y', 'distance': 2.8284},
                 {'name_gate1': 'crz', 'name_gate2': 'z', 'distance': 2.8284},
                 {'name_gate1': 'crz', 'name_gate2': 'cx', 'distance': 2.0},
                 {'name_gate1': 'crz', 'name_gate2': 'rx', 'distance': 2.29},
                 {'name_gate1': 'crz', 'name_gate2': 'ry', 'distance': 2.2878},
                 {'name_gate1': 'crz', 'name_gate2': 'rz', 'distance': 1.8004},
                 {'name_gate1': 'crz', 'name_gate2': 'crx', 'distance': 1.2749},
                 {'name_gate1': 'crz', 'name_gate2': 'cry', 'distance': 1.2739},
                 {'name_gate1': 'crz', 'name_gate2': 'crz', 'distance': 0}]


one_q_ops_name = ['h', 'rx', 'ry', 'rz', 'cx']
one_q_ops = [HGate, RXGate, RYGate, RZGate]
one_param_name = ['rx', 'ry', 'rz']
one_param = [RXGate, RYGate, RZGate]
two_param_name = ['u2']
two_param = [U2Gate]
two_q_ops = [CXGate]
three_param_name = ['u3']
three_param = [U3Gate]
three_q_ops_name = ['ccx']
three_q_ops = [CCXGate]
