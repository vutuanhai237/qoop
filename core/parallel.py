import qiskit
import typing
from ..backend import constant
from qiskit.primitives import Sampler


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
    sampler = Sampler()
    results = (
            sampler.run(qc)
            .result()
            .quasi_dists[0]
            .get(0, 0)
        )
    return results
