import typing
import qiskit
import random
import numpy as np
from ..backend import utilities

import numpy as np
import qiskit.quantum_info as qi

def create_params(depths, num_circuits, num_generations):
    # zip num_generations, depths, num_circuits as params
    params = []
    for num_generation in num_generations:
        for depth in depths:
            for num_circuit in num_circuits:
                params.append((depth, num_circuit, num_generation))
    return params        

def calculate_risk(utests, V):
    num_qubits = V.num_qubits
    # Create |0> state
    zero_state = np.zeros((2**num_qubits, 1))
    zero_state[0] = 1
    # Create |0><0| matrix
    zero_zero_dagger = np.outer(zero_state, np.conj(zero_state.T))
    V_matrix = qi.DensityMatrix(V).data
    risk = []
    for utest in utests:
        Ui_matrix = qi.DensityMatrix(utest).data
        # Eq inside L1 norm of matrix ^2
        eq = (Ui_matrix @ zero_zero_dagger @ np.conj(Ui_matrix.T) - V_matrix @ zero_zero_dagger @ np.conj(V_matrix.T))
        # L1 norm of matrix ^ 2
        risk.append(np.linalg.norm(eq, 1)**2)
    # Expected risk / 4
    return np.mean(risk)/4  

def fight(population):
    circuits = random.sample(population, 2)
    return circuits[0] if circuits[0].fitness > circuits[1].fitness else circuits[1]


def random_mutate(population, prob, mutate_func):
    random_circuit_index = np.random.randint(0, len(population))
    random_value = random.random()
    if random_value < prob:
        print(random_value)
        print(f'Mutate {random_circuit_index}')
        population[random_circuit_index].mutate(mutate_func)
    return population


def calculate_strength_point(self):
    inverse_fitnesss = [1 - circuit.fitness for circuit in self.population]
    mean_inverse_fitnesss = np.mean(inverse_fitnesss)
    std_inverse_fitnesss = np.std(inverse_fitnesss)
    strength_points = [(1 - circuit.fitness - mean_inverse_fitnesss) /
                        std_inverse_fitnesss for circuit in self.population]
    scaled_strength_points = utilities.softmax(strength_points, self.depth)
    for i, circuit in enumerate(self.population):
        circuit.strength_point = scaled_strength_points[i]
    return

def sort_by_fitness(objects: list, fitnesss: list):
    combined_list = list(zip(objects, fitnesss))
    sorted_combined_list = sorted(combined_list, key=lambda x: x[1], reverse=True)
    sorted_object = [item[0] for item in sorted_combined_list]
    return sorted_object