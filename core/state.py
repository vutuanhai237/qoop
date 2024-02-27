import qiskit
import numpy as np
import typing
import types
from ..backend import utilities
from scipy.linalg import expm
from qiskit.extensions import UnitaryGate
"""
Function to load classical data in a quantum device
This code is copy from https://github.com/adjs/dcsp/blob/master/encoding.py
"""


class bin_tree:
    size = None
    values = None

    def __init__(self, values):
        self.size = len(values)
        self.values = values

    def parent(self, key):
        return int((key - 0.5) / 2)

    def left(self, key):
        return int(2 * key + 1)

    def right(self, key):
        return int(2 * key + 2)

    def root(self):
        return 0

    def __getitem__(self, key):
        return self.values[key]


class Encoding:
    qcircuit = None
    quantum_data = None
    classical_data = None
    num_qubits = None
    tree = None
    output_qubits = []

    def __init__(self, input_vector, encode_type='amplitude_encoding'):
        if encode_type == 'amplitude_encoding':
            self.amplitude_encoding(input_vector)

    @staticmethod
    def _recursive_compute_beta(input_vector, betas):
        if len(input_vector) > 1:
            new_x = []
            beta = []
            for k in range(0, len(input_vector), 2):
                norm = np.sqrt(input_vector[k]**2 + input_vector[k + 1]**2)
                new_x.append(norm)
                if norm == 0:
                    beta.append(0)
                else:
                    if input_vector[k] < 0:
                        beta.append(
                            2 * np.pi - 2 *
                            np.arcsin(input_vector[k + 1] / norm))  # testing
                    else:
                        beta.append(2 * np.arcsin(input_vector[k + 1] / norm))
            Encoding._recursive_compute_beta(new_x, betas)
            betas.append(beta)

    @staticmethod
    def _index(k, circuit, control_qubits, numberof_controls):
        binary_index = '{:0{}b}'.format(k, numberof_controls)
        for j, qbit in enumerate(control_qubits):
            if binary_index[j] == '1':
                circuit.x(qbit)

    def amplitude_encoding(self, input_vector):
        """
        load real vector x to the amplitude of a quantum state
        """
        self.num_qubits = int(np.log2(len(input_vector)))
        self.quantum_data = qiskit.QuantumRegister(self.num_qubits)
        self.qcircuit = qiskit.QuantumCircuit(self.quantum_data)
        newx = np.copy(input_vector)
        betas = []
        Encoding._recursive_compute_beta(newx, betas)
        self._generate_circuit(betas, self.qcircuit, self.quantum_data)

    def _generate_circuit(self, betas, qcircuit, quantum_input):
        numberof_controls = 0  # number of controls
        control_bits = []
        for angles in betas:
            if numberof_controls == 0:
                qcircuit.ry(angles[0], quantum_input[self.num_qubits - 1])
                numberof_controls += 1
                control_bits.append(quantum_input[self.num_qubits - 1])
            else:
                for k, angle in enumerate(reversed(angles)):
                    Encoding._index(k, qcircuit, control_bits,
                                    numberof_controls)

                    qcircuit.mcry(angle,
                                  control_bits,
                                  quantum_input[self.num_qubits - 1 -
                                                numberof_controls],
                                  None,
                                  mode='noancilla')

                    Encoding._index(k, qcircuit, control_bits,
                                    numberof_controls)
                control_bits.append(quantum_input[self.num_qubits - 1 -
                                                  numberof_controls])
                numberof_controls += 1


def cf(qc: qiskit.QuantumCircuit, theta: float, qubit1: int, qubit2: int) -> qiskit.QuantumCircuit:
    """Add Controlled-F gate to quantum circuit

    Args:
        - qc (qiskit.QuantumCircuit): ddded circuit
        - theta (float): arccos(1/sqrt(num_qubits), base on number of qubit
        - qubit1 (int): control qubit
        - qubit2 (int): target qubit

    Returns:
        - qiskit.QuantumCircuit: Added circuit
    """
    cf = qiskit.QuantumCircuit(2)
    u = np.array([[1, 0, 0, 0], [0, np.cos(theta), 0,
                                 np.sin(theta)], [0, 0, 1, 0],
                  [0, np.sin(theta), 0, -np.cos(theta)]])
    cf.unitary(u, [0, 1])
    cf_gate = cf.to_gate(label='CF')
    qc.append(cf_gate, [qubit1, qubit2])
    return qc


def w3(circuit: qiskit.QuantumCircuit, qubit: int) -> qiskit.QuantumCircuit:
    """Create W state for 3 qubits

    Args:
        - circuit (qiskit.QuantumCircuit): added circuit
        - qubit (int): the index that w3 circuit acts on

    Returns:
        - qiskit.QuantumCircuit: added circuit
    """
    qc = qiskit.QuantumCircuit(3)
    theta = np.arccos(1 / np.sqrt(3))
    qc.cf(theta, 0, 1)
    qc.cx(1, 0)
    qc.ch(1, 2)
    qc.cx(2, 1)
    w3 = qc.to_gate(label='w3')
    # Add the gate to your circuit which is passed as the first argument to cf function:
    circuit.append(w3, [qubit, qubit + 1, qubit + 2])
    return circuit


qiskit.QuantumCircuit.w3 = w3
qiskit.QuantumCircuit.cf = cf


def w_sub(qc: qiskit.QuantumCircuit, num_qubits: int, shift: int = 0) -> qiskit.QuantumCircuit:
    """The below codes is implemented from [this paper](https://arxiv.org/abs/1606.09290)
    \n Simplest case: 3 qubits. <img src='../images/general_w.png' width = 500px/>
    \n General case: more qubits. <img src='../images/general_w2.png' width = 500px/>

    Args:
        - num_qubits (int): number of qubits
        - shift (int, optional): begin wire. Defaults to 0.

    Raises:
        - ValueError: When the number of qubits is not valid

    Returns:
        - qiskit.QuantumCircuit
    """
    if num_qubits < 2:
        raise ValueError('W state must has at least 2-qubit')
    if num_qubits == 2:
        # |W> state ~ |+> state
        qc.h(0)
        return qc
    if num_qubits == 3:
        # Return the base function
        qc.w3(shift)
        return qc
    else:
        # Theta value of F gate base on the circuit that it acts on
        theta = np.arccos(1 / np.sqrt(qc.num_qubits - shift))
        qc.cf(theta, shift, shift + 1)
        # Recursion until the number of qubits equal 3
        w_sub(qc, num_qubits - 1, qc.num_qubits - (num_qubits - 1))
        for i in range(1, num_qubits):
            qc.cnot(i + shift, shift)
    return qc


def ghz(num_qubits, theta: float = np.pi / 2) -> qiskit.QuantumCircuit:
    """Create GHZ state with a parameter

    Args:
        - num_qubits (int): number of qubits
        - theta (float): parameters

    Returns:
        - QuantumCircuit: the added circuit
    """
    if isinstance(theta, float) != True:
        theta = (theta['theta'])
    qc = qiskit.QuantumCircuit(num_qubits)
    qc.ry(theta, 0)
    for i in range(0, qc.num_qubits - 1):
        qc.cnot(0, i + 1)
    return qc


def ghz_inverse(num_qubits: int, theta: float = np.pi / 2) -> qiskit.QuantumCircuit:
    """Create GHZ state with a parameter

    Args:
        - qc (QuantumCircuit): Init circuit
        - theta (Float): Parameter

    Returns:
        - QuantumCircuit: the added circuit
    """
    if isinstance(theta, float) != True:
        theta = (theta['theta'])
    qc = qiskit.QuantumCircuit(num_qubits)
    for i in range(0, num_qubits - 1):
        qc.cnot(0, num_qubits - i - 1)
    qc.ry(-theta, 0)
    return qc

def specific_matrix(matrix: np.ndarray) -> qiskit.QuantumCircuit:
    if utilities.is_unitary(matrix) == False:
        matrix = utilities.to_unitary(matrix)
    num_qubits = int(np.log2(matrix.shape[0]))
    unitary_matrix = matrix
    unitary_gate = qiskit.QuantumCircuit(num_qubits)
    unitary_gate.unitary(unitary_matrix, list(range(0, num_qubits)))
    unitary_gate = unitary_gate.to_gate(label='InputUnita')
    qc = qiskit.QuantumCircuit(num_qubits)
    qc.append(unitary_gate, list(range(0, num_qubits)))
    return qc
def specific(state: np.ndarray) -> qiskit.QuantumCircuit:
    """Create a random Haar quantum state

    Args:
        num_qubits (int): number of qubits

    Returns:
        qiskit.QuantumCircuit
    """
    if isinstance(state, np.ndarray):
        num_qubits = int(np.log2(state.shape[0]))
    elif isinstance(state, list):
        num_qubits = int(np.log2(len(state)))    
    qc = qiskit.QuantumCircuit(num_qubits)
    if np.linalg.norm(state) != 1:
        import warnings
        warnings.warn("The input state is not normalized, we will normalize it for you")
    state = state / np.linalg.norm(state)
    qc.prepare_state(state, list(range(0, num_qubits)))
    return qc


def haar(num_qubits: int) -> qiskit.QuantumCircuit:
    """Create a random Haar quantum state

    Args:
        num_qubits (int): number of qubits

    Returns:
        qiskit.QuantumCircuit
    """
    psi = 2*np.random.rand(2**num_qubits)-1
    psi = psi / np.linalg.norm(psi)
    qc = qiskit.QuantumCircuit(num_qubits)
    qc.prepare_state(psi, list(range(0, num_qubits)))
    return qc


def haar_inverse(num_qubits: int) -> qiskit.QuantumCircuit:
    """Inverse version of haar random state

    Args:
        num_qubits (int): number of qubits

    Returns:
        qiskit.QuantumCircuit
    """
    psi = 2*np.random.rand(2**num_qubits)-1
    psi = psi / np.linalg.norm(psi)
    encoder = Encoding(psi, 'amplitude_encoding')
    qc = encoder.qcircuit
    return qc.inverse()


def w(num_qubits: int) -> qiskit.QuantumCircuit:
    """Create n-qubit W state based on the its number of qubits

    Args:
        - qc (qiskit.QuantumCircuit): init circuit

    Returns:
        - qiskit.QuantumCircuit
    """
    qc = qiskit.QuantumCircuit(num_qubits)
    qc.x(0)
    qc = w_sub(qc, qc.num_qubits)
    return qc


def w_inverse(qc: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    """Create inverse n-qubit W state based on the its number of qubits

    Args:
        - qc (qiskit.QuantumCircuit): init circuit

    Returns:
        - qiskit.QuantumCircuit
    """
    qc1 = qiskit.QuantumCircuit(qc.num_qubits)
    qc1.x(0)
    qc1 = w_sub(qc1, qc.num_qubits)
    qc = qc.combine(qc1.inverse())
    return qc


def ame(num_qubits: int) -> qiskit.QuantumCircuit:
    """Create n-qubit AME state based on the its number of qubits

    Args:
        - qc (qiskit.QuantumCircuit): init circuit

    Returns:
        - qiskit.QuantumCircuit
    """
    if num_qubits == 3:
        amplitude_state = np.array([
            0.27,
            0.363,
            0.326,
            0,
            0.377,
            0,
            0,
            0.740*(np.cos(-0.79*np.pi)+1j*np.sin(-0.79*np.pi))])
    elif num_qubits == 4:
        w = np.exp(2*np.pi*1j/3)
        amplitude_state = 1/np.sqrt(6)*np.array([
            0.0,
            0.0,
            1,
            0,
            0,
            w,
            w**2,
            0,
            0,
            w**2,
            w,
            0,
            1,
            0,
            0,
            0,])
    else:
        raise ValueError("The AME state currently available for 3 and 4 qubits")
    return specific(amplitude_state)


def ame_fake(num_qubits: int) -> qiskit.QuantumCircuit:
    """Create inverse n-qubit AME state based on the its number of qubits

    Args:
        - qc (qiskit.QuantumCircuit): init circuit

    Returns:
        - qiskit.QuantumCircuit
    """
    amplitude_state = np.array([
        0.27,
        0.363,
        0.326,
        0,
        0.377,
        0,
        0,
        np.sqrt(1-0.27**2-0.363**2-0.326**2-0.377**2)])
    qc = qiskit.QuantumCircuit(num_qubits)
    return specific(amplitude_state)


def calculate_hamiltonian(num_qubits):
    # pauli_x = np.array([[0,1],[1,0]])
    # pauli_z = np.array([[1,0],[0,-1]])
    # identity = np.array([[1,0],[0,1]])
    if num_qubits == 1:
        labels = ["X","Z"]
        coeffs = [1]*2
    if num_qubits == 2:
        labels = ["XI", "IX","ZZ","ZZ"]
        coeffs = [1]*4
    if num_qubits == 3:
        labels = ["XII", "IXI","IIX","ZZI","IZZ","ZIZ"]
        coeffs = [1]*6
    if num_qubits == 4:
        labels = ["XIII", "IXII","IIXI","IIIX","ZZII","IZZI","IIZZ","ZIIZ"]
        coeffs = [1,1,1,1,1,1,1,1]
    if num_qubits == 5:
        labels = ["XIIII", "IXIII","IIXII","IIIXI","IIIIX","ZZIII","IZZII","IIZZI","IIIZZ","ZIIIZ"]
        coeffs = [1]*10
    if num_qubits == 6:
        labels = ["XIIIII", "IXIIII","IIXIII","IIIXII","IIIIXI","IIIIIX","ZZIIII","IZZIII","IIZZII","IIIZZI","IIIIZZ","ZIIIIZ"]
        coeffs = [1]*12
    spo = qiskit.quantum_info.SparsePauliOp(labels,coeffs)
    return spo.to_matrix() 

# Calculating the eigenvalues and eigenvectors of the cost Hamiltonian

# Calculating the eigenvalues and eigenvectors of the cost Hamiltonian

def find_eigenvec_eigenval(matrix):

    value, vector = np.linalg.eig(matrix)
    new_vector = []
    for v in range(0, len(vector)):
        holder = []
        for h in range(0, len(vector)):
            holder.append(vector[h][v])
        new_vector.append(holder)

    return [value, np.array(new_vector)]

# Preaparing the partition function and each of the probability amplitudes of the diifferent terms in the TFD state

def calculate_terms_partition(eigenvalues,beta):

    list_terms = []
    partition_sum = 0
    for i in eigenvalues:
        list_terms.append(np.exp(-0.5*beta*i))
        partition_sum = partition_sum + np.exp(-1*beta*i)

    return [list_terms, np.sqrt(float(partition_sum.real))]

#Preparring the TFD state for the cost function

def construct_tfd_state(num_qubits,beta):

    # In this implementation, the eigenvectors of the Hamiltonian and the transposed Hamiltonian are calculated separately
    y_gate = 1
    y = np.array([[0, -1], [1, 0]])
    for i in range(0, num_qubits):
        y_gate = np.kron(y_gate, y)
    
    matrix = calculate_hamiltonian(num_qubits)
    eigen = find_eigenvec_eigenval(matrix)
    partition = calculate_terms_partition(eigen[0],beta)

    vec = np.zeros(2**(2*num_qubits))
    for i in range(0, 2**num_qubits):
        
        # time_rev = complex(0,1)*np.matmul(y_gate, np.conj(eigen[1][i]))
        addition = (float((partition[0][i]/partition[1]).real))*(np.kron(eigen[1][i], eigen[1][i]))
        vec = np.add(vec, addition)

    qc = qiskit.QuantumCircuit(num_qubits*2)
    amplitude_state = vec/np.sqrt(sum(np.absolute(vec) ** 2))
    qc.prepare_state(amplitude_state, list(range(0, num_qubits*2))) 
    return qiskit.compiler.transpile(qc,basis_gates=["h","cx","rx","ry",
    "rz","crx","cry","crz"],optimization_level=3)


def time_dependent_qc(num_qubits: int,h_opt, t):
    """create U circuit from h_opt and time t
    
    Args:
        - qc (QuantumCircuit): Init circuit
        - h_opt: Hamiltonian
        - t (float): time
        
    Returns:
        - QuantumCircuit: the added circuit
    """
    #Create circuit
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    
    # Ensure h_opt is Hermitian
    if not np.allclose(h_opt.to_matrix(), np.conj(h_opt.to_matrix()).T):
        raise ValueError("The Hamiltonian is not Hermitian.")

    # Calculate the unitary matrix
    U = expm(-1j * t * h_opt.to_matrix())

    # Check if U is unitary
    if not np.allclose(U @ U.conj().T, np.eye(U.shape[0])):
        raise ValueError("The resulting matrix U is not unitary.")

    # Create a UnitaryGate from the unitary_matrix
    unitary_gate = UnitaryGate(U)

    # Append the unitary_gate to the quantum circuit
    qc.append(unitary_gate, range(qc.num_qubits))

    return qc

def time_dependent_qc_inverse(num_qubits: int,h_opt, t):
    """create U circuit from h_opt and time t
    
    Args:
        - qc (QuantumCircuit): Init circuit
        - h_opt: Hamiltonian
        - t (float): time
        
    Returns:
        - QuantumCircuit: the added circuit
    """
    #Create circuit
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    
    # Ensure h_opt is Hermitian
    if not np.allclose(h_opt.to_matrix(), np.conj(h_opt.to_matrix()).T):
        raise ValueError("The Hamiltonian is not Hermitian.")

    # Calculate the unitary matrix
    U = expm(1j * t * h_opt.to_matrix())

    # Check if U is unitary
    if not np.allclose(U @ U.conj().T, np.eye(U.shape[0])):
        raise ValueError("The resulting matrix U is not unitary.")

    # Create a UnitaryGate from the unitary_matrix
    unitary_gate = UnitaryGate(U)

    # Append the unitary_gate to the quantum circuit
    qc.append(unitary_gate, range(qc.num_qubits))

    return qc