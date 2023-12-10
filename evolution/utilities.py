import typing
import qiskit
import random
import numpy as np

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