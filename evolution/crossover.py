
import numpy as np
import typing, types
from .ecircuit import ECircuit
from ..backend import utilities

def onepoint_crossover(circuit1: ECircuit, circuit2: ECircuit, 
                       percent: float = None, is_truncate=False) -> typing.Tuple:
    """Cross over between two circuits and create 2 offsprings

    Args:
        - circuit1 (evolution.ecircuit.ECircuit): Father
        - circuit2 (evolution.ecircuit.ECircuit): Mother
        - percent (float, optional): Percent of father's genome in offspring 1. Defaults to None.

    """
    # If percent is not produced, dividing base on how strong of father's fitness.
    standard_depth = circuit1.qc.depth()
    standard_fitness_func = circuit1.fitness_func
    strength_point_circuit1 = int(standard_depth * percent)
    sub11, sub12 = utilities.divide_circuit_by_depth(
        circuit1.qc, strength_point_circuit1)
    sub21, sub22 = utilities.divide_circuit_by_depth(
        circuit2.qc, strength_point_circuit1)
    combined_qc1 = utilities.compose_circuit([sub11, sub22])
    combined_qc2 = utilities.compose_circuit([sub21, sub12])
    if is_truncate:
        combined_qc1 = utilities.truncate_circuit(
            combined_qc1, standard_depth)
        combined_qc2 = utilities.truncate_circuit(
            combined_qc2, standard_depth)
    new_qc1 = ECircuit(combined_qc1, standard_fitness_func)
    new_qc2 = ECircuit(combined_qc2, standard_fitness_func)
    return new_qc1, new_qc2

def onepoint_crossover_strength_based(circuit1: ECircuit, circuit2: ECircuit, 
                       percent: float = None, is_truncate=False):
    """Cross over between two circuits and create 2 offsprings

    Args:
        circuit1 (evolution.ecircuit.ECircuit): Father
        circuit2 (evolution.ecircuit.ECircuit): Mother
        percent (float, optional): Percent of father's genome in offspring 1. Defaults to None.

    """
    # If percent is not produced, dividing base on how strong of father's fitness.
    standard_depth = circuit1.qc.depth()
    strength_point_circuit1 = np.round(circuit1.strength_point)
    standard_fitness_func = circuit1.fitness_func
    # if percent is None:
    #     percent = 1 - circuit1.fitness / (circuit1.fitness + circuit2.fitness)
    if percent is None:
        percent_point = strength_point_circuit1/standard_depth
    if percent_point > 0.99:
        combined_qc1, combined_qc2 = circuit1.qc.copy(), circuit1.qc.copy()
    else:
        sub11, sub12 = utilities.divide_circuit_by_depth(
            circuit1.qc, strength_point_circuit1)
        sub21, sub22 = utilities.divide_circuit_by_depth(
            circuit2.qc, strength_point_circuit1)
        combined_qc1 = utilities.compose_circuit([sub11, sub22])
        combined_qc2 = utilities.compose_circuit([sub21, sub12])
    if is_truncate:
        combined_qc1 = utilities.truncate_circuit(
            combined_qc1, standard_depth)
        combined_qc2 = utilities.truncate_circuit(
            combined_qc2, standard_depth)
    new_qc1 = ECircuit(combined_qc1, standard_fitness_func)
    new_qc2 = ECircuit(combined_qc2, standard_fitness_func)
    return new_qc1, new_qc2