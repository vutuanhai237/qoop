import numpy as np
import pennylane as qml
import pennylane.numpy as nps
import qiskit

dev = qml.device("default.qubit")

def circuit_curry(num_qubits, num_layers):
    # dev = qml.device("lightning.gpu", wires=num_qubits)
    @qml.qnode(dev)
    def circuit(thetas):
        k = 0
        for _ in range(num_layers):
            for i in range(0, num_qubits - 1):
                qml.CRY(thetas[k], wires = [i, i+1])
                k += 1
            qml.CRY(thetas[k], wires = [num_qubits - 1, 0])
            k += 1
            for i in range(0, num_qubits):
                qml.RX(thetas[k], wires = i)
                qml.RZ(thetas[k+1], wires = i)
                k += 2
        return qml.state()
    return circuit
    
def cost_curry(state, circuit):
    def cost_func(thetas):
        psi = nps.expand_dims((state), axis = 1)
        phi = nps.expand_dims(circuit(thetas), axis = 1)
        p0 = (nps.real(nps.dot(nps.conjugate(psi).T, phi))**2)[0][0]
        return 1 - p0
    return cost_func


def compilation(state, thetas, num_layers, steps = 100, opt = qml.AdamOptimizer(stepsize = 0.1)):
    # Thetas must be nps array
    num_qubits = int(np.log2(state.shape[0]))
    
    circuit = circuit_curry(num_qubits, num_layers)
    cost_func = cost_curry(state, circuit)
    grad_func = qml.grad(cost_func)

    costs = []
    thetass = []
    print(num_layers)
    if thetas is None:
        thetas = nps.array(np.ones(3*num_layers*num_qubits))

    for i in range(steps):
        thetas, prev_cost = opt.step_and_cost(cost_func, thetas, grad_fn = grad_func)
        costs.append(prev_cost)
        thetass.append(thetas)
    # print(i)
    del circuit
    return costs, thetass, i