import qiskit, typing, types, pickle, os, json
import qiskit.quantum_info as qi
import numpy as np
from ..core import metric, ansatz, state
from ..backend import utilities

class QuantumStatePreparation:
    def __init__(self, u: qiskit.QuantumCircuit | str, 
                 target_state: qiskit.QuantumCircuit | np.ndarray, 
                 thetas: np.ndarray = None):
        """There are four key atttributes for QSP problem: u, vdagger, parameters of u and name of u.

        Args:
            u (qiskit.QuantumCircuit | str): Ansatz
            target_state | vdagger (qiskit.QuantumCircuit | np.ndarray): Prepared state
            thetas (np.ndarray): Optimized parameters

        Returns:
            QuantumStatePreparation: completed object
        """
        self.num_layers = None
        self.thetas = thetas
        if isinstance(target_state, qiskit.QuantumCircuit):
            self.vdagger = target_state
        elif isinstance(target_state, np.ndarray):
            self.vdagger = state.specific(target_state)
        self.num_qubits = self.vdagger.num_qubits
        self.num_params = len(self.thetas)
        if isinstance(u, str):
            u_ansatz: typing.FunctionType = getattr(ansatz, u)
            basic_num_params = len(u_ansatz(self.num_qubits, 1).parameters)
            self.num_layers = int(self.num_params/basic_num_params)
            self.u = u_ansatz(self.num_qubits, self.num_layers)
        else:
            self.u = u

        traces, fidelities = metric.calculate_compilation_metrics(
            self.u, self.vdagger, np.expand_dims(self.thetas, axis=0))
        self.trace = traces[0]
        self.fidelity = fidelities[0]
        self.qc = self.u.bind_parameters(self.thetas)
        return
    
    def save(self, file_name: str):
        """Save QSP to .qspobj file with a given path

        Args:
            state (str): Name of vdagger
            file_name (str): Saved path
        """
        if not os.path.exists(file_name):
            os.mkdir(file_name)
        # If u is an outside ansatz, save it as qpy file
        path_to_u = os.path.join(file_name, "u")
        utilities.save_circuit(self.u, path_to_u)
        path_to_vdagger = os.path.join(file_name, "vdagger")
        utilities.save_circuit(self.vdagger, path_to_vdagger)
        qspobj = {
            'u': path_to_u,
            'vdagger': path_to_vdagger,
            'num_qubits': self.num_qubits,
            'num_layers': self.num_layers,
            'thetas': list(self.thetas),
        }
        with open(f"{os.path.join(file_name, 'info')}.json", "w") as file:
            json.dump(qspobj, file)
        return
    @staticmethod
    def load(file_name: str = ''):
        import pathlib
        file = pathlib.Path(file_name)
        if file.is_dir:
            qspjson = json.load(open(os.path.join(file_name, 'info.json')))
            qspobj = QuantumStatePreparation(
                u = utilities.load_circuit(qspjson['u']), 
                target_state= utilities.load_circuit(qspjson['vdagger']), 
                thetas = qspjson['thetas']
            )
        elif file.is_file:
            qspjson = json.load(open(file_name))
            target_state = getattr(state, qspjson['vdagger'])(qspjson['num_qubits'])
            # Only inverse() for existing state include: GHZ, W, AME, TFD
            qspobj = QuantumStatePreparation(u = qspjson['u'], target_state=target_state.inverse(), thetas = qspjson['thetas'])
        else:
            raise TypeError("Please input a path to json file or qsp folder")
        return qspobj
    @staticmethod
    def prepare(state: str | np.ndarray, error_rate: float = 0.01, **kwargs):
        if isinstance(state, str):
            return QuantumStatePreparation.prepare_existed(state, error_rate, kwargs['num_qubits'])
        elif isinstance(state, np.ndarray):
            return QuantumStatePreparation.prepare_random(state, error_rate, kwargs)
        else:
            raise TypeError("Please input target state name or an array")
    @staticmethod
    def prepare_random(state: np.ndarray, error_rate: float, **kwargs):
        
        return
    @staticmethod
    def prepare_existed(state: str, error_rate: float, num_qubits: int):
        """Find qspobj which satisfy all conditions

        Args:
            state (str): _description_
            num_qubits (int): _description_
            error_rate (float): _description_
            database_path (str): _description_

        Returns:
            typing.Union[None, QuantumStatePreparation]
        """
        module_dir = os.path.dirname(os.path.abspath(__file__))
        database_path = os.path.join(module_dir, 'qspjson/')
        best_qspobj = None
        files = [f for f in os.listdir(database_path) if os.path.isfile(os.path.join(database_path, f))]
        files = [f for f in files if f.split('_')[0] == state and int(f.split('_')[2]) == num_qubits]
        for i in range(0, len(files)):
            path = database_path + files[i]
            qspobj = QuantumStatePreparation.load_from_file(path)
            if qspobj.fidelity > 1 - error_rate:
                if best_qspobj is None:
                    best_qspobj = qspobj
                if qspobj.u.depth() < best_qspobj.u.depth():
                    best_qspobj = qspobj
                elif qspobj.u.depth() == best_qspobj.u.depth():
                    if qspobj.num_params < best_qspobj.num_params:
                        best_qspobj = qspobj
            
                # print(best_qspobj.fidelity,best_qspobj.u.depth(),best_qspobj.num_params)             
            
        if best_qspobj is None:
            print(f"Can not find the existing ansatz which can prepare state {state} {num_qubits} qubits >= {1 - error_rate} fidelity")
            return
        else:
            print(best_qspobj.u)
            print(f"can prepare the state {state} {num_qubits} qubits >= {1 - error_rate} fidelity ({best_qspobj.fidelity})")
            return best_qspobj


