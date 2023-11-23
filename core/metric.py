from ..core import ansatz, measure
from qiskit.primitives import Sampler
import numpy as np
import qiskit
import qiskit.quantum_info as qi
import scipy
import typing


def loss_basis(
    u: qiskit.QuantumCircuit,
    vdagger: qiskit.QuantumCircuit,
    thetass: typing.List[np.ndarray],
) -> typing.List:
    """Return loss value for loss function L = 1 - P_0
    \n Here P_0 ~ 1 or L ~ 0 will be the best value

    Args:
        - measurement_value (float): P_0 value

    Returns:
        - float: loss value
    """
    uvaddager: qiskit.QuantumCircuit | None = u.compose(vdagger)
    losss = []
    for thetas in thetass:
        measurement_value = measure.measure(uvaddager.copy(), thetas)
        losss.append(1 - measurement_value)
    return losss


def loss_fubini_study(
    u: qiskit.QuantumCircuit,
    vdagger: qiskit.QuantumCircuit,
    thetass: typing.List[np.ndarray],
) -> typing.List:
    """Return loss value for loss function C = (1 - P_0)^(1/2)
    \n Here P_0 ~ 1 or L ~ 0 will be the best value

    Args:
        - measurement_value (float): P_0 value

    Returns:
        - float: loss value
    """
    uvaddager: qiskit.QuantumCircuit | None = u.compose(vdagger)
    losss = []
    for thetas in thetass:
        measurement_value = measure.measure(uvaddager.copy(), thetas)
        losss.append(np.sqrt(1 - measurement_value))
    return losss


def calculate_ce_metric(u: qiskit.QuantumCircuit, exact=False) -> float:
    """calculate_concentratable_entanglement

    Args:
        - u (qiskit.QuantumCircuit): _description_
        - exact (bool, optional): _description_. Defaults to False.

    Returns:
        - float: ce value
    """
    qubit = list(range(u.num_qubits))
    n = len(qubit)
    cbits = qubit.copy()
    swap_test_circuit = ansatz.parallized_swap_test(u)

    if exact:
        statevec = qi.Statevector(swap_test_circuit)
        statevec.seed(value=42)
        probs = statevec.evolve(
            swap_test_circuit).probabilities_dict(qargs=qubit)
        return 1 - probs["0" * len(qubit)]
    else:
        for i in range(0, n):
            swap_test_circuit.measure(qubit[i], cbits[i])
        sampler = Sampler()
        result = sampler.run(
            swap_test_circuit).result().quasi_dists[0].get(0, 0)
        return 1 - result


def extract_state(qc: qiskit.QuantumCircuit) -> typing.Tuple:
    """Get infomation about quantum circuit

    Args:
        - qc (qiskit.QuantumCircuit): Extracted circuit

    Returns:
       - tuple: state vector and density matrix
    """
    psi = qi.Statevector.from_instruction(qc)
    rho_psi = qi.DensityMatrix(psi)
    return psi, rho_psi


def compilation_trace_distance(rho, sigma) -> float:
    """Since density matrices are Hermitian, so trace distance is 1/2 (Sigma(|lambdas|)) with lambdas are the eigenvalues of (rho_psi - rho_psi_hat) matrix

    Args:
        - rho (DensityMatrix): first density matrix
        - sigma (DensityMatrix): second density matrix

    Returns:
        - float: trace metric has value from 0 to 1
    """
    w, _ = np.linalg.eig((rho - sigma).data)
    return 1 / 2 * sum(abs(w))


def compilation_trace_fidelity(rho, sigma) -> float:
    """Calculating the fidelity metric

    Args:
        - rho (DensityMatrix): first density matrix
        - sigma (DensityMatrix): second density matrix

    Returns:
        - float: trace metric has value from 0 to 1
    """
    rho = rho.data
    sigma = sigma.data
    return np.real(
        np.trace(
            scipy.linalg.sqrtm((scipy.linalg.sqrtm(rho)) @ (rho))
            @ (scipy.linalg.sqrtm(sigma))
        )
    )


def gibbs_trace_fidelity(rho, sigma) -> float | None:
    """Calculating the fidelity metric

    Args:
        - rho (DensityMatrix): first density matrix
        - sigma (DensityMatrix): second density matrix

    Returns:
        - float: trace metric has value from 0 to 1
    """
    if rho is None:
        return None
    half_power_sigma = scipy.linalg.fractional_matrix_power(sigma, 1 / 2)
    return np.trace(scipy.linalg.sqrtm(half_power_sigma @ rho.data @ half_power_sigma))


def gibbs_trace_distance(rho) -> float | None:
    """Calculate trace distance

    Args:
        - rho (DensityMatrix)

    Returns:
        - float: trace metric has value from 0 to 1
    """
    if rho is None:
        return None
    return np.trace(np.linalg.matrix_power(rho, 2))


def calculate_premetric(
    u: qiskit.QuantumCircuit, vdagger: qiskit.QuantumCircuit, thetas: np.ndarray
) -> typing.Tuple:
    """For convinient

    Args:
        u (qiskit.QuantumCircuit)
        vdagger (qiskit.QuantumCircuit)
        thetas (np.ndarray)
        gibbs (bool, optional): Defaults to False.

    Returns:
        tuple: including binded qc, rho, sigma
    """
    if (len(u.parameters)) > 0:
        qc = u.assign_parameters(thetas)
        rho = qi.DensityMatrix(qc)
        sigma = qi.DensityMatrix(vdagger.inverse())

    else:
        qc = vdagger.assign_parameters(thetas).inverse()
        rho = qi.DensityMatrix(u)
        sigma = qi.DensityMatrix(qc)
    return qc, rho, sigma


def gibbs_metrics(
    u: qiskit.QuantumCircuit,
    vdagger: qiskit.QuantumCircuit,
    thetass: typing.List[np.ndarray],
) -> typing.Tuple | None:
    """Calculate gibbs metric through list of thetas

    Args:
        - u (qiskit.QuantumCircuit)
        - vdagger (qiskit.QuantumCircuit)
        - thetass (typing.List[np.ndarray]): list of parameters
        - gibbs (bool, optional): Defaults to False.

    Returns:
        - tuple: including gibbs_traces, gibbs_fidelities
    """
    gibbs_traces = []
    gibbs_fidelities = []
    for thetas in thetass:
        _, rho, sigma = calculate_premetric(u, vdagger, thetas)
        gibbs_rho = qi.partial_trace(rho, [0, 1])
        gibbs_sigma = qi.partial_trace(sigma, [0, 1])
        gibbs_trace = gibbs_trace_distance(gibbs_rho)
        gibbs_fidelity = gibbs_trace_fidelity(gibbs_rho, gibbs_sigma)
        gibbs_traces.append(gibbs_trace)
        gibbs_fidelities.append(gibbs_fidelity)
        return gibbs_traces, gibbs_fidelities


def gibbs_trace_distances(
    u: qiskit.QuantumCircuit,
    vdagger: qiskit.QuantumCircuit,
    thetass: typing.List[np.ndarray],
) -> typing.List:
    """Calculate gibbs metric through list of thetas

    Args:
        - u (qiskit.QuantumCircuit)
        - vdagger (qiskit.QuantumCircuit)
        - thetass (typing.List[np.ndarray]): list of parameters
        - gibbs (bool, optional): Defaults to False.

    Returns:
        - tuple: including gibbs_traces, gibbs_fidelities
    """
    gibbs_traces = []
    for thetas in thetass:
        _, rho, sigma = calculate_premetric(u, vdagger, thetas)
        gibbs_rho = qi.partial_trace(rho, [0, 1])
        gibbs_sigma = qi.partial_trace(sigma, [0, 1])
        gibbs_trace = gibbs_trace_distance(gibbs_rho)
        gibbs_traces.append(gibbs_trace)
    return gibbs_traces


def gibbs_trace_fidelities(
    u: qiskit.QuantumCircuit,
    vdagger: qiskit.QuantumCircuit,
    thetass: typing.List[np.ndarray],
) -> typing.List:
    gibbs_fidelities = []
    for thetas in thetass:
        _, rho, sigma = calculate_premetric(u, vdagger, thetas)
        gibbs_rho = qi.partial_trace(rho, [0, 1])
        gibbs_sigma = qi.partial_trace(sigma, [0, 1])
        gibbs_fidelity = gibbs_trace_fidelity(gibbs_rho, gibbs_sigma)
        gibbs_fidelities.append(gibbs_fidelity)
    return gibbs_fidelities


def ces(
    u: qiskit.QuantumCircuit,
    vdagger: qiskit.QuantumCircuit,
    thetass: typing.List[np.ndarray],
) -> typing.List:
    """Calculate CE metric through list of thetas

    Args:
        - u (qiskit.QuantumCircuit)
        - vdagger (qiskit.QuantumCircuit)
        - thetass (typing.List[np.ndarray])

    Returns:
        - List: list of ce value
    """
    ces = []
    for thetas in thetass:
        qc, _, _ = calculate_premetric(u, vdagger, thetas)
        ces.append(calculate_ce_metric(qc))
    return ces


def compilation_trace_fidelities(
    u: qiskit.QuantumCircuit,
    vdagger: qiskit.QuantumCircuit,
    thetass: typing.List[np.ndarray],
) -> typing.List:
    """Calculate compilation metric through list of thetas

    Args:
        - u (qiskit.QuantumCircuit)
        - vdagger (qiskit.QuantumCircuit)
        - thetass (List[np.ndarray]): list of parameters

    Returns:
        - fidelities
    """
    compilation_fidelities = []
    for thetas in thetass:
        _, rho, sigma = calculate_premetric(u, vdagger, thetas)
        compilation_fidelity = compilation_trace_fidelity(rho, sigma)
        compilation_fidelities.append(compilation_fidelity)
    return compilation_fidelities


def compilation_metrics(
    u: qiskit.QuantumCircuit,
    vdagger: qiskit.QuantumCircuit,
    thetass: typing.List[np.ndarray],
) -> typing.Tuple:
    """Calculate compilation metric through list of thetas

    Args:
        - u (qiskit.QuantumCircuit)
        - vdagger (qiskit.QuantumCircuit)
        - thetass (List[np.ndarray]): list of parameters

    Returns:
        - tuple: including traces and fidelities
    """
    compilation_traces = []
    compilation_fidelities = []
    for thetas in thetass:
        _, rho, sigma = calculate_premetric(u, vdagger, thetas)
        compilation_trace = compilation_trace_distance(rho, sigma)
        compilation_fidelity = compilation_trace_fidelity(rho, sigma)
        compilation_traces.append(compilation_trace)
        compilation_fidelities.append(compilation_fidelity)
    return compilation_traces, compilation_fidelities


def compilation_trace_distances(
    u: qiskit.QuantumCircuit,
    vdagger: qiskit.QuantumCircuit,
    thetass: typing.List[np.ndarray],
) -> typing.List:
    """Calculate compilation metric through list of thetas

    Args:
        - u (qiskit.QuantumCircuit)
        - vdagger (qiskit.QuantumCircuit)
        - thetass (List[np.ndarray]): list of parameters

    Returns:
        - traces
    """
    compilation_traces = []
    for thetas in thetass:
        _, rho, sigma = calculate_premetric(u, vdagger, thetas)
        compilation_trace = compilation_trace_distance(rho, sigma)
        compilation_traces.append(compilation_trace)
    return compilation_traces
