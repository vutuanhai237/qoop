import qiskit
from .divider import by_depth as divider_by_depth
from .divider import by_num_cnot as divider_by_num_cnot

def by_num_cnot(num_cnot: int) -> qiskit.QuantumCircuit:
    """Crop circuit until achieve desired number of CNOT gates

    Args:
        - qc (qiskit.QuantumCircuit)
        - selected_num_cnot (int)

    Returns:
        - qiskit.QuantumCircuit: Truncated circuit
    """
    def by_num_cnot_func(qc: qiskit.QuantumCircuit):
        qc1, _ = (divider_by_num_cnot(num_cnot))(qc)
        return qc1
    return by_num_cnot_func
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
            qc1, _ = (divider_by_depth(depth))(qc)
            return qc1
    return by_depth_func