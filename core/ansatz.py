import qiskit
import math
import random
import numpy as np
from ..backend import constant, utilities
from qiskit.circuit import ParameterVector

def graph(num_qubits: int) -> qiskit.QuantumCircuit:
    """Create parameterized graph ansatz

    Args:
        - num_qubits (int):

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits)
    edges = constant.edges_graph_state[num_qubits]
    thetas = ParameterVector('theta', len(edges))
    i = 0
    for edge in edges:
        control_bit = int(edge.split('-')[0])
        controlled_bit = int(edge.split('-')[1])
        qc.crz(thetas[i], control_bit, controlled_bit)
        i += 1
    return qc


def stargraph(num_qubits: int, num_layers: int = 1) -> qiskit.QuantumCircuit:
    """Create parameterized star-graph ansatz

    Args:
        - num_qubits (int)
        - num_layers (int, optional): Defaults to 1.

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    thetas = ParameterVector(
        'theta', num_layers * (2 * num_qubits - 2))
    qc = qiskit.QuantumCircuit(num_qubits)
    j = 0
    for l in range(0, num_layers, 1):
        for i in range(0, num_qubits):
            qc.ry(thetas[j], i)
            j += 1
        qc.cz(0, 1)
        for i in range(2, num_qubits):
            qc.ry(thetas[j], 0)
            j += 1
            qc.cz(0, i)
    return qc


def polygongraph(num_qubits: int = 3, num_layers: int = 1) -> qiskit.QuantumCircuit:
    """Create parameterized polygon-graph ansatz

    Args:
        - num_qubits (int)
        - num_layers (int, optional): Defaults to 1.

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = ParameterVector('theta', 2*num_qubits*num_layers)
    j = 0
    for _ in range(0, num_layers, 1):
        for i in range(0, num_qubits):
            qc.ry(thetas[j], i)
            j += 1
        for i in range(0, num_qubits - 1, 2):
            qc.cz(i, i + 1)
        if num_qubits % 2 == 1:
            for i in range(0, num_qubits - 1):
                qc.ry(thetas[j], i)
                j += 1
        else:
            for i in range(0, num_qubits):
                qc.ry(thetas[j], i)
                j += 1
        for i in range(1, num_qubits - 1, 2):
            qc.cz(i, i + 1)
        if num_qubits % 2 == 1:
            qc.ry(thetas[j], num_qubits - 1)
            j += 1
        qc.cz(0, num_qubits - 1)
        qc.barrier()
    return qc


def hadamard_hypergraph(num_qubits: int, num_layers: int = 1) -> qiskit.QuantumCircuit:
    """Create parameterized hadamard hyper-graph ansatz

    Args:
        - num_qubits (int)
        - num_layers (int, optional): Defaults to 1.

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = ParameterVector('theta', 3*num_qubits*num_layers)
    j = 0
    for i in range(0, num_qubits):
        qc.h(i)
    for _ in range(0, num_layers, 1):
        for i in range(0, num_qubits):
            qc.ry(thetas[j], i)
            j += 1
        qc.cz(0, 1)
        for i in range(0, num_qubits - 1):
            qc.ry(thetas[j], i)
            j += 1
        qc.cz(0, num_qubits - 1)
        qc.ry(thetas[j], 0)
        j += 1
        qc.ry(thetas[j], num_qubits - 1)
        j += 1
        qc.cz(num_qubits - 2, num_qubits - 1)
        for i in range(1, num_qubits):
            qc.ry(thetas[j], i)
            j += 1
        qc.ccz(0, 1, 2)
    return qc


def hypergraph(num_qubits: int, num_layers: int = 1) -> qiskit.QuantumCircuit:
    """Create parameterized hyper-graph ansatz

    Args:
        - num_qubits (int)
        - num_layers (int, optional): Defaults to 1.

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = ParameterVector('theta', 3*num_qubits*num_layers)
    j = 0
    for _ in range(0, num_layers, 1):
        for i in range(0, num_qubits):
            qc.ry(thetas[j], i)
            j += 1
        qc.cz(0, 1)
        for i in range(0, num_qubits - 1):
            qc.ry(thetas[j], i)
            j += 1
        qc.cz(0, num_qubits - 1)
        qc.ry(thetas[j], 0)
        j += 1
        qc.ry(thetas[j], num_qubits - 1)
        j += 1
        qc.cz(num_qubits - 2, num_qubits - 1)
        for i in range(1, num_qubits):
            qc.ry(thetas[j], i)
            j += 1
        qc.ccz(0, 1, 2)
    return qc


def hypergraph_zxz(num_qubits: int = 3, num_layers: int = 1) -> qiskit.QuantumCircuit:
    """Create parameterized hyper-graph zxz ansatz

    Args:
        - num_qubits (int)
        - num_layers (int, optional): Defaults to 1.

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = utilities.compose_circuit(
            [qc, hypergraph(num_qubits), zxz_layer(num_qubits)])
    return qc


def entangled_layer(qc: qiskit.QuantumCircuit):
    """Create entanglement state

    Args:
        - qc (QuantumCircuit): init circuit

    Returns:
        - QuantumCircuit: parameterized quantum circuit
    """
    for i in range(0, qc.num_qubits):
        if i == qc.num_qubits - 1:
            qc.cnot(qc.num_qubits - 1, 0)
        else:
            qc.cnot(i, i + 1)
    return qc


def cry_layer(num_qubits: int) -> qiskit.QuantumCircuit:
    """Create Control - RY layer

    Args:
        - num_qubits (int)

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits)
    thetas = ParameterVector('theta', num_qubits)
    for i in range(0, qc.num_qubits - 1, 2):
        qc.cry(thetas[i], i, i + 1)
    for i in range(1, qc.num_qubits - 1, 2):
        qc.cry(thetas[i], i, i + 1)
    qc.cry(thetas[qc.num_qubits - 1], qc.num_qubits - 1, 0)
    return qc


def binho(num_qubits: int = 3, num_layers: int = 1) -> qiskit.QuantumCircuit:
    """Create binho ansatz

    Args:
        - num_qubits (int, optional): Defaults to 3.
        - num_layers (int, optional): Defaults to 1.

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = utilities.compose_circuit([qc, rx_layer(num_qubits),
                                        cry_layer(num_qubits).inverse(),
                                        rz_layer(num_qubits),
                                        cry_layer(num_qubits).inverse(),
                                        rz_layer(num_qubits)])
    return qc


def ry_layer(num_qubits: int = 3, shift=0) -> qiskit.QuantumCircuit:
    """Create RY layer

    Args:
        - num_qubits (int, optional): Defaults to 3.
        - shift (int, optional): Defaults to 0.

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = ParameterVector('theta', num_qubits - shift)
    for i in range(0, num_qubits):
        qc.ry(thetas[i], i + shift)
    return qc


def swap_layer(num_qubits: int = 3, shift=0) -> qiskit.QuantumCircuit:
    """Create SWAP layer

    Args:
        - num_qubits (int, optional): Defaults to 3.
        - shift (int, optional): Defaults to 0.

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for i in range(0 + shift, qc.num_qubits - 1, 2):
        qc.swap(i, i + 1)
    return qc


def alternating_ZXZlayer(num_qubits: int = 3,
                                num_layers: int = 1) -> qiskit.QuantumCircuit:
    """Create Alternating ZXZ layer

    Args:
        - num_qubits (int, optional): Defaults to 3.
        - num_layers (int, optional): Defaults to 1.

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = utilities.compose_circuit([qc, ry_layer(num_qubits),
                                        swap_layer(num_qubits),
                                        ry_layer(num_qubits),
                                        swap_layer(num_qubits, shift=1),
                                        ry_layer(num_qubits),
                                        swap_layer(num_qubits),
                                        ry_layer(num_qubits),
                                        swap_layer(num_qubits, shift=1),
                                        ry_layer(num_qubits)])
    return qc


###########################
#### Tomography ansatz  ###
###########################

def chain_zxz_pennylane(num_qubits: int, num_layers):
    import pennylane as qml
    dev = qml.device("default.qubit", wires=num_qubits)
    @qml.qnode(dev, diff_method="parameter-shift")
    def chain_zxz_pennylane_func(thetas):
        j = 0
        for _ in range(0, num_layers):
            for i in range(0, num_qubits - 1):
                qml.CRY(thetas[j], wires=[i,i+1])
                j += 1
            qml.CRY(thetas[j], wires=[num_qubits - 1, 0])
            j += 1
            for i in range(0, num_qubits):
                qml.RZ(thetas[j], wires=i)
                qml.RX(thetas[j+1], wires=i)
                qml.RZ(thetas[j+2], wires=i)
                j += 3
        return
    return chain_zxz_pennylane_func


def Wchain(num_qubits: int) -> qiskit.QuantumCircuit:
    """Create W_chain ansatz

    Args:
        - num_qubits (int)

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = ParameterVector('theta', num_qubits)
    for i in range(0, num_qubits - 1):
        qc.cry(thetas[i], i, i + 1)
    qc.cry(thetas[-1], num_qubits - 1, 0)
    return qc

def WchainCNOT(num_qubits: int) -> qiskit.QuantumCircuit:
    """Create W_chain ansatz

    Args:
        - num_qubits (int)

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = ParameterVector('theta', num_qubits)
    for i in range(0, num_qubits - 1):
        qc.cx(i, i + 1)
    qc.cx(num_qubits - 1, 0)
    return qc

def Walternating(num_qubits: int, index_layer: int) -> qiskit.QuantumCircuit:
    """Create Walternating ansatz

    Args:
        - num_qubits (int)
        - index_layer (int)

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    def calculate_n_walternating(num_qubits: int, index_layers: int) -> int:
        """calculate number of parameter in walternating base on index_layer

        Args:
            num_qubits (int)
            index_layers (int)

        Returns:
            int
        """
        if index_layers % 2 == 0:
            n_walternating = int(num_qubits / 2)
        else:
            n_walternating = math.ceil(num_qubits / 2)
        return n_walternating

    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = ParameterVector(
        'theta', calculate_n_walternating(num_qubits, index_layer))
    t = 0
    if index_layer % 2 == 0:
        # Even
        for i in range(1, qc.num_qubits - 1, 2):
            qc.cry(thetas[t], i, i + 1)
            t += 1
        qc.cry(thetas[-1], 0, qc.num_qubits - 1)
    else:
        # Odd
        for i in range(0, qc.num_qubits - 1, 2):
            qc.cry(thetas[t], i, i + 1)
            t += 1
    return qc

def WalternatingCNOT(num_qubits: int, index_layer: int) -> qiskit.QuantumCircuit:
    """Create Walternating ansatz

    Args:
        - num_qubits (int)
        - index_layer (int)

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    def calculate_n_walternating(num_qubits: int, index_layers: int) -> int:
        """calculate number of parameter in walternating base on index_layer

        Args:
            num_qubits (int)
            index_layers (int)

        Returns:
            int
        """
        if index_layers % 2 == 0:
            n_walternating = int(num_qubits / 2)
        else:
            n_walternating = math.ceil(num_qubits / 2)
        return n_walternating

    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = ParameterVector(
        'theta', calculate_n_walternating(num_qubits, index_layer))
    t = 0
    if index_layer % 2 == 0:
        # Even
        for i in range(1, qc.num_qubits - 1, 2):
            qc.cx(i, i + 1)
            t += 1
        qc.cx(0, qc.num_qubits - 1)
    else:
        # Odd
        for i in range(0, qc.num_qubits - 1, 2):
            qc.cx(i, i + 1)
            t += 1
    return qc

def Walltoall(num_qubits: int, limit: int=0) -> qiskit.QuantumCircuit:
    """Create Walternating ansatz

    Args:
        - num_qubits (int)
        - limit (int): Default to 0

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    def calculate_n_walltoall(num_qubits: int) -> int:
        """Calculate the number of Walltoall

        Args:
            - num_qubits (int):

        Returns:
            - int
        """
        n_walltoall = 0
        for i in range(1, num_qubits):
            n_walltoall += i
        return n_walltoall

    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = ParameterVector(
        'theta', calculate_n_walltoall(num_qubits))
    if limit == 0:
        limit = len(thetas)
    t = 0
    for i in range(0, num_qubits):
        for j in range(i + 1, num_qubits):
            qc.cry(thetas[t], i, j)
            t += 1
            if t == limit:
                return qc
    return qc


def WalltoallCNOT(num_qubits: int, limit=0) -> qiskit.QuantumCircuit:
    """Create Walltoall CNOT

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - limit (int): limit layer

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits)
    t = 0
    for i in range(0, qc.num_qubits):
        for j in range(i + 1, qc.num_qubits):
            qc.cnot(i, j)
            t += 1
            if t == limit:
                return qc
    return qc

def Wchain_xyz(num_qubits: int, num_layers: int = 1) -> qiskit.QuantumCircuit:
    """Create Wchain + XYZ
    Args:
        - num_qubits (int)
        - num_layers (int, optional): Defaults to 1.

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = utilities.compose_circuit([qc, Wchain(num_qubits),
                                        xyz_layer(num_qubits)])
    return qc

def Wchain_zxz(num_qubits: int, num_layers: int = 1) -> qiskit.QuantumCircuit:
    """Create Wchain + ZXZ
    Args:
        - num_qubits (int)
        - num_layers (int, optional): Defaults to 1.

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = utilities.compose_circuit([qc, Wchain(num_qubits),
                                        zxz_layer(num_qubits)])
    return qc


def Walternating_zxz(num_qubits, num_layers: int = 1, index_layer: int = 0) -> qiskit.QuantumCircuit:
    """Create Walternating + ZXZ
    Args:
        - num_qubits (int)
        - num_layers (int, optional): Defaults to 1.

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = utilities.compose_circuit([qc, Walternating(num_qubits, index_layer),
                                        zxz_layer(num_qubits)])
    return qc


def Walltoall_zxz(num_qubits: int, num_layers: int = 1, limit=0) -> qiskit.QuantumCircuit:
    """Create Walltoall + ZXZ
    Args:
        - num_qubits (int)
        - num_layers (int, optional): Defaults to 1.
        - limit (int): limit number of parameter
    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = utilities.compose_circuit(
            [qc, Walltoall(num_qubits, limit=limit), zxz_layer(num_qubits)])
    return qc

def WchainCNOT_xyz(num_qubits: int = 3,
                                  num_layers: int = 1) -> qiskit.QuantumCircuit:
    """Create WalltoallCNOT + ZXZ

    Args:
        - num_qubits (int, optional): efaults to 3.
        - num_layers (int, optional): Defaults to 1.
        - limit (int, optional): Defaults to 0.

    Returns:
        qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = utilities.compose_circuit(
            [qc, WchainCNOT(num_qubits), xyz_layer(num_qubits)])
    return qc

def zxz_WchainCNOT(num_qubits: int = 3,
                                  num_layers: int = 1) -> qiskit.QuantumCircuit:
    """Create WalltoallCNOT + ZXZ

    Args:
        - num_qubits (int, optional): efaults to 3.
        - num_layers (int, optional): Defaults to 1.
        - limit (int, optional): Defaults to 0.

    Returns:
        qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = utilities.compose_circuit(
            [qc, zxz_layer(num_qubits), WchainCNOT(num_qubits)])
    return qc

def zxz_WalternatingCNOT(num_qubits: int = 3,
                                  num_layers: int = 1, index_layer: int = 0) -> qiskit.QuantumCircuit:
    """Create WalltoallCNOT + ZXZ

    Args:
        - num_qubits (int, optional): efaults to 3.
        - num_layers (int, optional): Defaults to 1.
        - limit (int, optional): Defaults to 0.

    Returns:
        qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = utilities.compose_circuit(
            [qc, zxz_layer(num_qubits), WalternatingCNOT(num_qubits, index_layer)])
    return qc

def zxz_WalltoallCNOT(num_qubits: int = 3,
                                  num_layers: int = 1,
                                  limit=0) -> qiskit.QuantumCircuit:
    """Create WalltoallCNOT + ZXZ

    Args:
        - num_qubits (int, optional): efaults to 3.
        - num_layers (int, optional): Defaults to 1.
        - limit (int, optional): Defaults to 0.

    Returns:
        qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits)
    for _ in range(0, num_layers):
        qc = utilities.compose_circuit(
            [qc, zxz_layer(num_qubits), WalltoallCNOT(num_qubits, limit=limit)])
    return qc


def WalltoallCNOT_zxz(num_qubits: int = 3,
                                  num_layers: int = 1,
                                  limit=0) -> qiskit.QuantumCircuit:
    """Create WalltoallCNOT + ZXZ

    Args:
        - num_qubits (int, optional): efaults to 3.
        - num_layers (int, optional): Defaults to 1.
        - limit (int, optional): Defaults to 0.

    Returns:
        qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = utilities.compose_circuit(
            [qc, WalltoallCNOT(qc, limit=limit), zxz_layer(num_qubits)])
    return qc

def xyz_layer(num_qubits: int = 3, num_layers: int = 1) -> qiskit.QuantumCircuit:
    """XYZ layer

    Args:
        - num_qubits (int, optional): Defaults to 3.
        - num_layers (int, optional): Defaults to 1.

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = utilities.compose_circuit(
            [qc, rx_layer(num_qubits), ry_layer(num_qubits), rz_layer(num_qubits)])
    return qc

def zxz_layer(num_qubits: int = 3, num_layers: int = 1) -> qiskit.QuantumCircuit:
    """ZXZ layer

    Args:
        - num_qubits (int, optional): Defaults to 3.
        - num_layers (int, optional): Defaults to 1.

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = utilities.compose_circuit(
            [qc, rz_layer(num_qubits), rx_layer(num_qubits), rz_layer(num_qubits)])
    return qc


def random_ccz(num_qubits: int, num_gates: int) -> qiskit.QuantumCircuit:
    """Adds a random number of CZ or CCZ gates (up to `max_gates`) to the given circuit.

    Args:
        - num_qubits (int)
        - num_gates (int)

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    sure = True if num_gates < 3 else False
    for _ in range(num_gates):
        if np.random.randint(2, size=1) == 0 or sure:
            wires = random.sample(range(0, num_qubits), 2)
            qc.cz(wires[0], wires[1])
        else:
            wires = random.sample(range(0, num_qubits), 3)
            wires.sort()
            qc.ccz(wires[0], wires[1], wires[2])
    return qc


def rz_layer(num_qubits: int) -> qiskit.QuantumCircuit:
    """RZ layer

    Args:
        - num_qubits (int)

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = ParameterVector('theta', num_qubits)
    for i in range(num_qubits):
        qc.rz(thetas[i], i)
    return qc


def rx_layer(num_qubits: int) -> qiskit.QuantumCircuit:
    """RX layer

    Args:
        - num_qubits (int)

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = ParameterVector('theta', num_qubits)
    for i in range(num_qubits):
        qc.rx(thetas[i], i)
    return qc


def ry_layer(num_qubits: int) -> qiskit.QuantumCircuit:
    """RY layer

    Args:
        - num_qubits (int)

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = ParameterVector('theta', num_qubits)
    for i in range(num_qubits):
        qc.ry(thetas[i], i)
    return qc


def cz_layer(num_qubits: int) -> qiskit.QuantumCircuit:
    """CZ layer

    Args:
        - num_qubits (int)

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    return qiskit.circuit.library.MCMT('z', num_qubits - 1, 1)


def g2(num_qubits: int, num_layers: int) -> qiskit.QuantumCircuit:
    """g2 ansatz

    Args:
        - num_qubits (int)
        - num_layers (int)

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = ParameterVector(
        'theta', 2 * num_qubits * num_layers)
    j = 0
    for _ in range(num_layers):
        for i in range(num_qubits):
            qc.ry(thetas[j], i)
            j += 1
        for i in range(0, num_qubits - 1, 2):
            qc.cz(i, i + 1)
        for i in range(num_qubits):
            qc.ry(thetas[j], i)
            j += 1
        for i in range(1, num_qubits - 1, 2):
            qc.cz(i, i + 1)
        qc.cz(0, num_qubits - 1)
    return qc


def gn(num_qubits: int, num_layers: int) -> qiskit.QuantumCircuit:
    """gn ansatz

    Args:
        - num_qubits (int)
        - num_layers (int)

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = utilities.compose_circuit(
            [qc, ry_layer(num_qubits), cz_layer(num_qubits)])
    return qc


def g2gn(num_qubits: int, num_layers: int) -> qiskit.QuantumCircuit:
    """g2 + gn ansatz

    Args:
        - num_qubits (int)
        - num_layers (int)

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = utilities.compose_circuit(
            [qc, g2(num_qubits, 1), gn(num_qubits, 1)])
    return qc


def g2gnw(num_qubits: int, num_layers: int) -> qiskit.QuantumCircuit:
    """g2 + gn + w ansatz

    Args:
        - num_qubits (int)
        - num_layers (int)

    Returns:
        - qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = utilities.compose_circuit([qc, g2(num_qubits, 1), gn(
            num_qubits, 1), zxz_layer(num_qubits, 1)])
    return qc


def parallized_swap_test(u: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    """Create multiple-qubit swap test

    Args:
        - u (qiskit.QuantumCircuit): init circuit

    Returns:
        - qiskit.QuantumCircuit: added circuit
    """
    n_qubit = u.num_qubits
    qubits_list_first = list(range(n_qubit, 2*n_qubit))
    qubits_list_second = list(range(2*n_qubit, 3*n_qubit))

    # Create swap test circuit
    swap_test_circuit = qiskit.QuantumCircuit(3*n_qubit, n_qubit)

    # Add initial circuit the first time

    swap_test_circuit = swap_test_circuit.compose(u, qubits=qubits_list_first)
    # Add initial circuit the second time
    swap_test_circuit = swap_test_circuit.compose(u, qubits=qubits_list_second)
    swap_test_circuit.barrier()

    # Add hadamard gate
    swap_test_circuit.h(list(range(0, n_qubit)))
    swap_test_circuit.barrier()

    for i in range(n_qubit):
        # Add control-swap gate
        swap_test_circuit.cswap(i, i+n_qubit, i+2*n_qubit)
    swap_test_circuit.barrier()

    # Add hadamard gate
    swap_test_circuit.h(list(range(0, n_qubit)))
    swap_test_circuit.barrier()
    return swap_test_circuit
