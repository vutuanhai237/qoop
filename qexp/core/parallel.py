import qiskit
import typing
from ..backend import constant
def measure(qcs: typing.List[qiskit.QuantumCircuit], qubits, cbits=[]):
    """Measuring the quantu circuit which fully measurement gates

    Args:
        - qc (QuantumCircuit): Measured circuit
        - qubits (np.ndarray): List of measured qubit

    Returns:
        - float: Frequency of 00.. cbit
    """
    n = len(qubits)
    if cbits == []:
        cbits = qubits.copy()
    for qc in qcs:
        if qc.num_clbits == 0:
            cr = qiskit.ClassicalRegister(qc.num_qubits, 'c')
            qc.add_register(cr)
        for i in range(0, n):
            qc.measure(qubits[i], cbits[i])
    counts = qiskit.execute(
            qcs, backend=constant.backend,
            shots=constant.NUM_SHOTS).result().get_counts()
    results = []
    for count in counts:
        results.append(count.get("0" * len(qubits), 0) / constant.NUM_SHOTS)
    return results