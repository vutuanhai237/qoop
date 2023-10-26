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



