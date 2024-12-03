import qiskit
from .divider import by_depth


def by_depth(depth: int) -> qiskit.QuantumCircuit:
    """Crop circuit until achieve desired depth value

    Args:
        - qc (qiskit.QuantumCircuit)
        - selected_depth (int)

    Returns:
        - qiskit.QuantumCircuit: Truncated circuit
    """
    def by_depth_func(qc: qiskit.QuantumCircuit):
        if qc.depth() <= depth:
            return qc
        else:
            qc1, _ = by_depth(qc, depth)
            return qc1
    return by_depth_func