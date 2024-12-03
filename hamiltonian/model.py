
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import SparsePauliOp

def ising(N,J,h):
    """create Hamiltonian for Ising model
        H = J * sum_{i = 0}^{N-1) Z_i*Z_{i+1} + h*sum_{i=0}^{N-1}X_i
    """

    pauli_op_list = []  
    for j in range(N):
        s = ('I' * (N - j - 1)) + 'X' + ('I' * j)
        pauli_op_list.append((s, h))
 
    for j in range(N-1):
        s = ('I' * (N - j - 2)) + 'ZZ' + ('I' * j)
        pauli_op_list.append((s, J))
        
    h_op = SparsePauliOp.from_list(pauli_op_list)

    return h_op

def XYZ_model(N,J,u,h):
    """create Hamiltonian for Ising model
        H = J * sum_{i = 0}^{N-1) Z_i*Z_{i+1} + h*sum_{i=0}^{N-1}X_i
    """

    pauli_op_list = []  
    for j in range(N):
        s = ('I' * (N - j - 1)) + 'Z' + ('I' * j)
        pauli_op_list.append((s, h[j]))
 
    for j in range(N-1):
        s = ('I' * (N - j - 2)) + 'ZZ' + ('I' * j)
        pauli_op_list.append((s, u))
    
    for j in range(N-1):
        s = ('I' * (N - j - 2)) + 'XX' + ('I' * j) + ('I' * (N - j - 2)) + 'YY' + ('I' * j)
        pauli_op_list.append((s, -J))
        
    h_op = SparsePauliOp.from_list(pauli_op_list)

    return h_op

def pauli_oper(N, oper = None):
    
    pauli_op_list = []  
    s = ('I' * (N - 1)) + oper
    pauli_op_list.append((s, 1.0))
        
    h_op = SparsePauliOp.from_list(pauli_op_list)
    return h_op    

def pauli_collect(N, oper = None):
    
    pauli_op_list = []  
    for j in range(N):
        s = ('I' * (N - j - 1)) + oper + ('I' * j)
        pauli_op_list.append((s, 1.0))
        
    h_op = SparsePauliOp.from_list(pauli_op_list)
    return h_op   