import qiskit
import typing

def by_num_cnot(qc: qiskit.QuantumCircuit, k: int) -> typing.List[qiskit.QuantumCircuit]:
    """Dividing circuit into two sub-circuits based on the number of CNOT gates

    Args:
        - qc (qiskit.QuantumCircuit)
        - k (int): number of CNOT gates in the first sub-circuit

    Returns:
        - typing.List[qiskit.QuantumCircuit]: two separated quantum circuits
    """
    if k < 0:
        raise ValueError("The number of CNOT gates must be >= 0")

    qc1 = qiskit.QuantumCircuit(qc.num_qubits)
    qc2 = qiskit.QuantumCircuit(qc.num_qubits)
    cnot_count = 0
    stop = 0

    for i in range(len(qc)):
        if qc[i][0].name == 'cx':
            cnot_count += 1
        qc1.append(qc[i][0], qc[i][1])
        stop += 1
        if cnot_count == k and i + 1 < len(qc) and qc[i+1][0].name == 'cx':
            for x in qc[stop:]:
                qc2.append(x[0], x[1])
            return qc1, qc2

    return qc1, qc2


def by_percent_depth(qc: qiskit.QuantumCircuit, percent: float) -> typing.List[qiskit.QuantumCircuit]:
    """Dividing circuit into two sub-circuits

    Args:
        - qc (qiskit.QuantumCircuit)
        - percent (float): from 0 to 1

    Returns:
        - typing.List[qiskit.QuantumCircuit]: two seperated quantum circuits
    """

    qc1 = qiskit.QuantumCircuit(qc.num_qubits)
    qc2 = qc1.copy()
    stop = 0
    for x in qc:
        qc1.append(x[0], x[1])
        stop += 1
        if qc1.depth() / qc.depth() >= percent:
            for x in qc[stop:]:
                qc2.append(x[0], x[1])
            return qc1, qc2
    return qc1, qc2

def by_half_depth(qc: qiskit.QuantumCircuit) -> typing.List[qiskit.QuantumCircuit]:
    """Dividing circuit into two sub-circuits

    Args:
        - qc (qiskit.QuantumCircuit)

    Returns:
        - typing.List[qiskit.QuantumCircuit]: two seperated quantum circuits
    """
    standard_depth = qc.depth()
    return by_percent_depth(qc, int(standard_depth/2))

def by_depth(depth: int) -> typing.List[qiskit.QuantumCircuit]:
    """Dividing circuit into two sub-circuits

    Args:
        - qc (qiskit.QuantumCircuit)
        - depth (int): specific depth value

    Returns:
        - typing.List[qiskit.QuantumCircuit]: two seperated quantum circuits
    """
    def by_depth_func(qc: qiskit.QuantumCircuit):
        def look_forward(qc, x):
            qc.append(x[0], x[1])
            return qc
        standard_depth = qc.depth()
        qc1 = qiskit.QuantumCircuit(qc.num_qubits)
        qc2 = qc1.copy()
        if depth < 0:
            raise "The depth must be >= 0"
        elif depth == 0:
            qc2 = qc.copy()
        elif depth == standard_depth:
            qc1 = qc.copy()
        else:
            stop = 0
            for i in range(len(qc)):
                qc1.append(qc[i][0], qc[i][1])
                stop += 1
                if qc1.depth() == depth and i + 1 < len(qc) and look_forward(qc1.copy(), qc[i+1]).depth() > depth:
                    for x in qc[stop:]:
                        qc2.append(x[0], x[1])
                    return qc1, qc2

    return by_depth_func