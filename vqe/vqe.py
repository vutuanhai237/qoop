import qiskit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.units import DistanceUnit
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.primitives import Estimator
from qiskit_algorithms.utils import algorithm_globals
from qiskit.algorithms.optimizers import SLSQP

def general_VQE(qc: qiskit.QuantumCircuit, atom: str, basis: str):
    driver = PySCFDriver(
        atom=atom,
        basis=basis,
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    problem = driver.run()
    hamiltonian = problem.hamiltonian.second_q_op()
    optimizer = SLSQP(maxiter=1000)
    estimator = Estimator()
    algorithm_globals.random_seed = 50
    mapper=JordanWignerMapper()
    qubit_op = mapper.map(hamiltonian)
    vqe = VQE(estimator = estimator, ansatz = qc, optimizer=optimizer)
    computation_value = vqe.compute_minimum_eigenvalue(qubit_op).eigenvalue.real
    return computation_value


V = V_optimal
risks = []
for u in u_test:
    risk = abc(V)
    risks.append(risk)

risk = np.mean(risks) * 1/4
=> ríks =?

=> thetas_optimal_C

Tính risk

=> thetas_optimal_R1

