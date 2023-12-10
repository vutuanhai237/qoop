import typing
import qiskit
import scipy
import tqdm
import numpy as np
from qiskit import qpy
from ..backend import constant


class EarlyStopping:
    def __init__(self, patience=0, delta=0):
        """Class for early stopper

        Args:
            patience (int, optional): The number of unchanged loss values step. Defaults to 0.
            delta (int, optional): Minimum distance between loss and a better loss. Defaults to 0.
        """
        self.patience = patience
        self.delta = delta
        self.wait = 0
        self.mode = "inactive"
        self.counter = 0

    def set_mode(self, mode: str):
        """Declare mode

        Args:
            mode (str): Assigned mode
        """
        self.mode = mode

    def get_mode(self) -> str:
        """Get mode

        Returns:
            str: Mode
        """
        return self.mode

    def track(self, old_loss: float, new_loss: float) -> None:
        """If new loss does not change, we will active early stopping.

        Args:
            old_loss (float)
            new_loss (float)
        """
        if self.mode == "inactive":
            if new_loss >= old_loss - self.delta:
                if self.counter == 0:
                    self.mode = "active"
                    self.counter = self.patience
                else:
                    self.counter -= 1
            return

    def invest(self, old_loss: float, new_loss: float) -> bool:
        """Detect small change

        Args:
            old_loss (float)
            new_loss (float)

        Returns:
            bool: 
        """
        if new_loss < old_loss - self.delta / 10:
            self.mode = "inactive"
            return True
        else:
            return False

# Copy from stackoverflow
class ProgressBar(object):
    def __init__(self, max_value, disable=True):
        self.max_value = max_value
        self.disable = disable
        self.p = self.pbar()

    def pbar(self):
        return tqdm.tqdm(total=self.max_value,
                         desc='Step: ',
                         disable=self.disable)

    def update(self, update_value):
        self.p.update(update_value)

    def close(self):
        self.p.close()


def get_wires_of_gate(gate: qiskit.circuit.Gate) -> typing.List[int]:
    """Get index bit that gate act on

    Args:
        - gate (qiskit.circuit.Gate): Quantum gate

    Returns:
        - List[int]: list of index bits
    """
    list_wire = []
    for register in gate[1]:
        list_wire.append(register.index)
    return list_wire


def is_gate_in_list_wires(gate: qiskit.circuit.Gate, wires: typing.List[int]) -> bool:
    """Check if a gate lies on the next layer or not

    Args:
        - gate (qiskit.circuit.Gate): Quantum gate
        - wires (typing.List[int]): list of index bits

    Returns:
        - Bool
    """
    list_wire = get_wires_of_gate(gate)
    for wire in list_wire:
        if wire in wires:
            return True
    return False


def split_into_layers(qc: qiskit.QuantumCircuit) -> typing.List:
    """Split a quantum circuit into layers

    Args:
        - qc (qiskit.QuantumCircuit): origin circuit

    Returns:
        - list: list of list of quantum gates
    """
    layers = []
    layer = []
    wires = []
    is_param_layer = None
    for gate in qc.data:
        name = gate[0].name
        if name in constant.ignore_generator:
            continue
        param = gate[0].params
        wire = get_wires_of_gate(gate)
        if is_param_layer is None:
            if len(param) == 0:
                is_param_layer = False
            else:
                is_param_layer = True
        # New layer's condition: depth increase or convert from non-parameterized layer to parameterized layer or vice versa
        if is_gate_in_list_wires(gate, wires) or (is_param_layer == False and len(param) != 0) or (is_param_layer == True and len(param) == 0):
            if is_param_layer == False:
                # First field is 'Is parameterized layer or not?'
                layers.append((False, layer))
            else:
                layers.append((True, layer))
            layer = []
            wires = []
        # Update sub-layer status
        if len(param) == 0 or name == 'state_preparation_dg':
            is_param_layer = False
        else:
            is_param_layer = True
        for w in wire:
            wires.append(w)
        layer.append((name, param, wire))
    # Last sub-layer
    if is_param_layer == False:
        # First field is 'Is parameterized layer or not?'
        layers.append((False, layer))
    else:
        layers.append((True, layer))
    return layers


def create_observers(qc: qiskit.QuantumCircuit, k: int = 0) -> typing.List:
    """Return dictionary of observers

    Args:
        - qc (qiskit.QuantumCircuit): Current circuit
        - k (int, optional): Number of observers each layer. Defaults to qc.num_qubits.

    Returns:
        - List
    """
    if k == 0:
        k = qc.num_qubits
    observer = []
    for gate in (qc.data)[-k:]:
        gate_name = gate[0].name
        # Non-param gates
        if gate_name in ['barrier', 'swap']:
            continue
        # 2-qubit param gates
        if gate[0].name in ['crx', 'cry', 'crz', 'cx', 'cz']:
            # Take controlled wire as index
            wire = qc.num_qubits - 1 - gate[1][1].index
            # Take control wire as index
            # wire = qc.num_qubits - 1 - gate[1][0].index
        # Single qubit param gates
        else:
            wire = qc.num_qubits - 1 - gate[1][0].index
        observer.append([gate_name, wire])
    return observer


def get_cry_index(qc: qiskit.QuantumCircuit, thetas: np.ndarray):
    """Return a list where i_th = 1 mean thetas[i] is parameter of CRY gate

    Args:
        - func (types.FunctionType): The creating circuit function
        - thetas (np.ndarray): Parameters
    Returns:
        - np.ndarray: The index list has length equal with number of parameters
    """
    layers = split_into_layers(qc)
    index_list = []
    for layer in layers:
        for gate in layer[1]:
            if gate[0] == 'cry':
                index_list.append(1)
            else:
                index_list.append(0)
            if len(index_list) == len(thetas):
                return index_list
    return index_list


def add_layer_into_circuit(qc: qiskit.QuantumCircuit, layer: typing.List) -> qiskit.QuantumCircuit:
    """Based on information in layers, adding new gates into current circuit

    Args:
        - qc (qiskit.QuantumCircuit): calculating circuit
        - layer (list): list of gate's informations

    Returns:
        - qiskit.QuantumCircuit: added circuit
    """
    for name, param, wire in layer:
        if name == 'rx':
            qc.rx(param[0], wire[0])
        if name == 'ry':
            qc.ry(param[0], wire[0])
        if name == 'rz':
            qc.rz(param[0], wire[0])
        if name == 'crx':
            qc.crx(param[0], wire[0], wire[1])
        if name == 'cry':
            qc.cry(param[0], wire[0], wire[1])
        if name == 'crz':
            qc.crz(param[0], wire[0], wire[1])
        if name == 'cz':
            qc.cz(wire[0], wire[1])
    return qc

def save_circuit(qc: qiskit.QuantumCircuit, file_name: str) -> None:
    """Save circuit as qpy object

    Args:
        qc (qiskit.QuantumCircuit): Saved circuit
        file_name (str): Path
    """

    with open(f"{file_name}.qpy", "wb") as qpy_file_write:
        qpy.dump(qc, qpy_file_write)
    return


def load_circuit(file_name: str) -> qiskit.QuantumCircuit:
    """Load circuit from a specific path

    Args:
        file_name (str): Path

    Returns:
        qiskit.QuantumCircuit
    """
    with open(f"{file_name}.qpy", "rb") as qpy_file_read:
        qc = qpy.load(qpy_file_read)[0]
    return qc

def unit_vector(i: int, length: int) -> np.ndarray:
    """Create vector where a[i] = 1 and a[j] = 0 with j <> i

    Args:
        - i (int): index
        - length (int): dimensional of vector

    Returns:
        - np.ndarray
    """
    vector = np.zeros((length))
    vector[i] = 1.0
    return vector


def haar_measure(n):
    """A Random matrix distributed with Haar measure

    Args:
        - n: dimensional of matrix
    """
    z = (scipy.randn(n, n) + 1j*scipy.randn(n, n))/scipy.sqrt(2.0)
    q, r = scipy.linalg.qr(z)
    d = scipy.diagonal(r)
    ph = d/scipy.absolute(d)
    q = scipy.multiply(q, ph, q)
    return q


def normalize_matrix(matrix: np.ndarray):
    """Follow the formula from Bin Ho

    Args:
        - matrix (numpy.ndarray): input matrix

    Returns:
        - np.ndarray: normalized matrix
    """
    return np.conjugate(np.transpose(matrix)) @ matrix / np.trace(np.conjugate(np.transpose(matrix)) @ matrix)


def softmax(xs: np.ndarray, max_value: float) -> np.ndarray:
    """Scale all elements in array from 0 to max_value

    Args:
        - xs (np.ndarray): array
        - max_value (float)

    Returns:
        - np.ndarray
    """
    exp_xs = np.exp(xs)
    return 1 + max_value * exp_xs / np.sum(exp_xs)


def is_pos_def(matrix: np.ndarray, error=1e-8) -> bool:
    """Check if a matrix is positive define matrix

    Args:
        - matrix (np.ndarray)
        - error (float, optional): error rate. Defaults to 1e-8.

    Returns:
        - bool: if True, matrix is positive define.
    """
    return np.all(np.linalg.eigvalsh(matrix) > -error)


def is_normalized(matrix: np.ndarray) -> bool:
    """Check if a matrix is normalized

    Args:
        - matrix (np.ndarray)

    Returns:
        - bool
    """
    return np.isclose(np.trace(matrix), 1)


def truncate_circuit(qc: qiskit.QuantumCircuit, selected_depth: int) -> qiskit.QuantumCircuit:
    """Crop circuit until achieve desired depth value

    Args:
        - qc (qiskit.QuantumCircuit)
        - selected_depth (int)

    Returns:
        - qiskit.QuantumCircuit: Truncated circuit
    """
    if qc.depth() <= selected_depth:
        return qc
    else:
        qc1, _ = divide_circuit_by_depth(qc, selected_depth)
        return qc1


def divide_circuit(qc: qiskit.QuantumCircuit, percent: float) -> typing.List[qiskit.QuantumCircuit]:
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


def divide_circuit_by_depth(qc: qiskit.QuantumCircuit, depth: int) -> typing.List[qiskit.QuantumCircuit]:
    """Dividing circuit into two sub-circuits

    Args:
        - qc (qiskit.QuantumCircuit)
        - depth (int): specific depth value

    Returns:
        - typing.List[qiskit.QuantumCircuit]: two seperated quantum circuits
    """
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
    return qc1, qc2


def remove_last_gate(qc: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    """Remove last gate

    Args:
        - qc (qiskit.QuantumCircuit)

    Returns:
        - qiskit.QuantumCircuit: Circuit which is removed last gate.
    """
    if qc.data:
        qc.data.pop()
    return qc


def compose_circuit(qcs: typing.List[qiskit.QuantumCircuit]) -> qiskit.QuantumCircuit:
    """Combine list of paramerterized quantum circuit into one. It's very helpful!!!

    Args:
        - qcs (typing.List[qiskit.QuantumCircuit]): List of quantum circuit

    Returns:
        - qiskit.QuantumCircuit: composed quantum circuit
    """
    qc = qiskit.QuantumCircuit(qcs[0].num_qubits)
    i = 0
    num_params = 0
    for sub_qc in qcs:
        num_params += len(sub_qc.parameters)
    thetas = qiskit.circuit.ParameterVector('theta', num_params)
    for sub_qc in qcs:
        for instruction in sub_qc:
            if len(instruction[0].params) == 1:
                instruction[0].params[0] = thetas[i]
                i += 1
            if len(instruction[0].params) == 3:
                instruction[0].params[0] = thetas[i:i+1]
                i += 2
            qc.append(instruction[0], instruction[1])
    return qc
def find_nearest(xs: typing.List, x):
    xs = np.asarray(xs)
    idx = (np.abs(xs - x)).argmin()
    return xs[idx]

def append_to_dict(tuple, new_items):
    for key, value in new_items.items():
        # Check if the key already exists in the dictionary
        if key in tuple:
            # If the key exists, append the value to its list
            tuple[key].append(value)
        else:
            # If the key doesn't exist, create a new key-value pair with the key and a list containing the value
            tuple[key] = [value]
            
def is_unitary(matrix: np.ndarray) -> bool:
    """Check if a matrix is unita or not

    Args:
        - matrix (np.ndarray): Input matrix

    Returns:
        - bool: Is unita or not
    """
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False
    # Check if U^\dagger U is approximately equal to the identity matrix
    product = np.dot(np.conjugate(matrix).T, matrix)
    identity_matrix = np.eye(matrix.shape[0])

    # Tolerance for numerical precision
    tolerance = 1e-10
    return np.allclose(product, identity_matrix)

def to_unitary(matrix: np.ndarray) -> np.ndarray:
    """Convert arbitrary matrix to unitary matrix

    Args:
        - matrix (np.ndarray): Input matrix

    Raises:
        - ValueError: Can not convert matrix to unitary matrix

    Returns:
        - np.ndarray: Unitary matrix
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    normalized_eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)
    if is_unitary(normalized_eigenvectors):
        return normalized_eigenvectors
    else:
        raise ValueError("This matrix can not convert to unitary form!")
        return 0
    
def to_state(dict: typing.Dict) -> np.ndarray:
    """Convert a dict example: dict = {'000': 1/np.sqrt(2), '001': 1j/np.sqrt(6), '100': 0.5} to state vector
    
    Output: [0.70710678+0.j         0.        +0.40824829j 0.        +0.j
            0.        +0.j         0.5       +0.j         0.        +0.j
            0.        +0.j         0.        +0.j        ]
    Args:
        - dict (typing.Dict): Input dict

    Returns:
        - np.ndarray: State vector
    """
    n = len(list(dict.keys())[0])
    state = np.zeros(2**n, dtype=np.complex128)
    for key, value in dict.items():
        index = int(key, 2)
        state[index] = value
    return state
