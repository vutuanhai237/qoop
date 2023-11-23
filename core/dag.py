
import qiskit
from ..backend import constant
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from ..backend import constant
torch.manual_seed(1000)
torch.cuda.manual_seed(1000)
np.random.seed(1000)
random.seed(1000)

class NodeDAG:
    def __init__(self, node: qiskit.dagcircuit.DAGOpNode):
        self.id = id(node)
        self.op_name = node.op.name
        self.index = 0
        self.successor = []
    def set_index(self, index):
        self.index = index
    def set_successor(self, successor):
        self.successor = successor

class TinyDAG():
    def __init__(self, dag: qiskit.dagcircuit.DAGCircuit):
        self.dag = dag
        self.nodes = []
        for index, node in enumerate(self.dag.op_nodes()):
            tiny_node = NodeDAG(node)
            tiny_node.set_index(index)
            self.nodes.append(tiny_node)
    def find_node(self, id):
        for i in range(len(self.nodes)):
            if (self.nodes[i].id == id): 
                return self.nodes[i]
        return None
    def construct(self):
        for node in (self.dag.op_nodes()): 
            sucessors = []
            for node_in in self.dag.quantum_successors(node):
                sucessor = self.find_node(id(node_in))
                if sucessor is not None:
                    sucessors.append(sucessor)
            self.nodes[self.find_node(id(node)).index].successor = sucessors
        return

def circuit_to_adjacency_matrix(qc: qiskit.QuantumCircuit):
    dag = qiskit.converters.circuit_to_dag(qc)
    tinyDAG = TinyDAG(dag)
    tinyDAG.construct()
    dag_size = qc.num_qubits * qc.depth()
    adjacency_matrix = np.zeros((dag_size, dag_size), dtype=np.float64)
    for i in range(len(tinyDAG.nodes)):
        for node in tinyDAG.nodes[i].successor:
            distance_info = None
            for d in constant.gate_distaces:
                if d["name_gate1"] == tinyDAG.nodes[i].op_name and d["name_gate2"] == node.op_name:
                    distance_info = d
            adjacency_matrix[tinyDAG.nodes[i].index][node.index] = distance_info['distance']
    return adjacency_matrix

def convert_string_to_int(string):
    return sum([ord(char) - 65 for char in string])


def circuit_to_dag(qc):
    """Convert a circuit to graph.
    Read more: 
    - https://qiskit.org/documentation/retworkx/dev/tutorial/dags.html
    - https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.commutation_dag.html

    Args:
        qc (qiskit.QuantumCircuit): A qiskit quantum circuit

    Returns:
        DAG: direct acyclic graph
    """
    return qml.transforms.commutation_dag(qml.from_qiskit(qc))()


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.matmul(adj, x)
        return x

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolution(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GraphConvolution(hidden_dim, hidden_dim))
        self.layers.append(GraphConvolution(hidden_dim, output_dim))

    def forward(self, x, adj):
        for layer in self.layers:
            x = F.relu(layer(x, adj))
        return x
    
def dag_to_node_features(dag):
    node_features = []
    for i in range(dag.size):
        node = dag.get_node(i)
        operation = constant.look_up_operator[node.op.base_name]
        params = node.op.parameters
        if len(params) == 0:
            params = [0]
        node_features.append([convert_string_to_int(operation), *params])
    return np.array(node_features)

def dag_to_adjacency_matrix(dag):
    num_nodes = dag.size
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(dag.size):
        node = dag.get_node(i)
        for successor in node.successors:
            adjacency_matrix[i][successor] = 1
    return np.array(adjacency_matrix)


def graph_to_scalar(node_features, adjacency_matrix):
    num_nodes = node_features.shape[0]
    input_dim = node_features.shape[1]
    hidden_dim = 64
    output_dim = 1
    num_layers = 2
    # Convert node features and adjacency matrix to tensors
    node_features = torch.tensor(node_features, dtype=torch.float32)
    adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
    # Create GCN model
    gcn = GCN(input_dim, hidden_dim, output_dim, num_layers)
    # Forward pass through the model
    graph_embedding = gcn(node_features, adjacency_matrix)
    # Apply global sum pooling to obtain a scalar representation
    # Apply global sum pooling to obtain a scalar representation
    graph_scalar = torch.sum(graph_embedding)
    # Sigmod activation to ranged value from 0 to 1
    return 1 / (1 + np.exp(-graph_scalar.item()))

def circuit_to_scalar(qc: qiskit.QuantumCircuit)->float:
    """Evaluate circuit

    Args:
        qc (qiskit.QuantumCircuit): encoded circuit

    Returns:
        float: Value from 0 to 1
    """
    dag = circuit_to_dag(qc)
    node_features = dag_to_node_features(dag)
    adjacency_matrix = dag_to_adjacency_matrix(dag)
    return graph_to_scalar(node_features, adjacency_matrix)