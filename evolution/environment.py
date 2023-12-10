

from .ecircuit import ECircuit
from .selection import sastify_circuit
from ..evolution import crossover, mutate, selection, threshold
from ..core import random_circuit
from ..backend import utilities
from dataclasses import dataclass, field
import types
import typing
import random
import os
import json
import datetime
import pathlib
import qiskit
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures


def extract_fitness(circuits: typing.List[ECircuit]):
    extracted_score = []
    for circuit in circuits:
        extracted_score.append(circuit.fitness)
    return extracted_score


def extract_circuit(circuits: typing.List[ECircuit]):
    extracted_circuits = []
    for circuit in circuits:
        extracted_circuits.append(circuit)
    return extracted_circuits


def bypass_compile(circuit: ECircuit):
    circuit.compile()
    print('Bypass', circuit.fitness)
    return circuit


def multiple_compile(circuits: typing.List[ECircuit]):
    executor = concurrent.futures.ProcessPoolExecutor()
    results = executor.map(bypass_compile, circuits)
    return results


@dataclass
class EEnvironmentMetadata:
    num_qubits: int
    depth: int
    num_circuit: int
    num_generation: int
    current_generation: int = field(default_factory=lambda: 0)
    fitnessss: list = field(default_factory=lambda: [])
    prob_mutate: float = field(default_factory=lambda: 0.1)


class EEnvironment():
    """Saved information for evolution process
    """

    def __init__(self, metadata: EEnvironmentMetadata | dict,
                 fitness_func: types.FunctionType = None,
                 crossover_func: types.FunctionType = crossover.onepoint_crossover,
                 mutate_func: types.FunctionType = mutate.bitflip_mutate,
                 selection_func: types.FunctionType = selection.elitist_selection,
                 threshold_func: types.FunctionType = threshold.compilation_threshold,
                 ) -> None:
        """_summary_

        Args:
            params (typing.Union[typing.List, str]): Other params for GA proces
            fitness_func (types.FunctionType, optional): Defaults to None.
            crossover_func (types.FunctionType, optional): Defaults to None.
            mutate_func (types.FunctionType, optional): Defaults to None.
            selection_func (types.FunctionType, optional): Defaults to None.
            pool (_type_, optional): Pool gate. Defaults to None.
            file_name (str, optional): Path of saved file.
        """

        self.metadata = metadata
        self.fitness_func = fitness_func
        self.crossover_func = crossover_func
        self.mutate_func = mutate_func
        self.selection_func = selection_func
        self.threshold_func = threshold_func
        if isinstance(metadata, EEnvironmentMetadata):
            self.metadata = metadata
        elif isinstance(metadata, dict):
            self.metadata = EEnvironmentMetadata(
                num_qubits=metadata.get('num_qubits', 0),
                depth=metadata.get('depth', 0),
                num_circuit=metadata.get('num_circuit', 0),
                num_generation=metadata.get('num_generation', 0),
                current_generation=metadata.get('current_generation', 0),
                fitnessss=metadata.get('fitnessss', []),
                prob_mutate=metadata.get('prob_mutate', []),
            )
        self.fitnesss: list = []
        self.circuits: typing.List[ECircuit] = []
        self.circuitss: typing.List[typing.List[ECircuit]] = []
        self.best_circuit = None
        self.best_fitness = 0
        return

    def set_fitness_func(self, fitness_func):
        self.fitness_func = fitness_func
        return

    def set_circuitss(self, circuitss):
        self.circuitss: typing.List[typing.List[qiskit.QuantumCircuit]] = circuitss
        return

    def set_circuits(self, circuits):
        self.circuits: typing.List[qiskit.QuantumCircuit] = circuits
        return

    def evol(self, verbose: int = 0):
        if verbose == 1:
            bar = utilities.ProgressBar(
                max_value=self.metadata.num_generation, disable=False)
        if self.metadata.current_generation == 0:
            print("Initialize list of circuit ...")
            self.init()
            print("Start evol progress ...")
        elif self.metadata.current_generation == self.metadata.num_generation:
            return
        else:
            print(
                f"Continute evol progress at generation {self.metadata.current_generation} ...")
        for generation in range(self.metadata.current_generation, self.metadata.num_generation):
            #####################
            ##### Pre-process ###
            #####################
            self.metadata.current_generation += 1
            print(f"Evol at generation {self.metadata.current_generation}")
            self.fitnesss = []
            #####################
            ######## Cost #######
            #####################
            # new_population = multiple_compile(new_population)
            for i in range(len(self.circuits)):
                self.fitnesss.append(self.fitness_func(self.circuits[i]))
            print(self.fitnesss)
            if generation > 0:
                self.metadata.fitnessss.append(self.fitnesss)
            #####################
            #### Threshold ######
            #####################
            if self.best_circuit is None or self.fitness_func(self.best_circuit) < np.max(self.fitnesss):
                self.best_circuit = self.circuits[np.argmax(self.fitnesss)]
                self.best_fitness = np.max(self.fitnesss)
                if self.threshold_func(self.best_fitness):
                    print(
                        f'End evol progress soon at generation {self.metadata.current_generation}, best score ever: {self.best_fitness}')
                    return

            #####################
            ##### Selection #####
            #####################
            self.circuits = self.selection_func(self.circuits, self.fitnesss)
            # random.shuffle(self.circuits)
            #####################
            ##### Cross-over ####
            #####################
            new_circuits: typing.List[ECircuit] = []
            for i in range(0, int(self.metadata.num_circuit/2), 2):
                # print("Cross-over")
                # print(self.circuits[i].draw())
                # print(self.circuits[i + 1].draw())
                offspring1, offspring2 = self.crossover_func(
                    self.circuits[i], self.circuits[i + 1])
                # print("Child")
                # print(offspring1.draw())
                # print(offspring2.draw())
                new_circuits.extend(
                    [self.circuits[i], self.circuits[i + 1], offspring1, offspring2])

            ####################
            ##### Mutation #####
            ####################
            for i in range(0, len(new_circuits)):
                if random.random() < self.metadata.prob_mutate:
                    print('Mutate')
                    new_circuits[i] = self.mutate_func(new_circuits[i])
            self.circuits = new_circuits
            self.circuitss.append(new_circuits)
            # self.save(self.file_name + f'ga_{self.num_qubits}qubits_{self.fitness_func.__name__}_{datetime.datetime.now().strftime("%Y-%m-%d")}.envobj')
            if verbose == 1:
                bar.update(1)
            if verbose == 2 and generation % 5 == 0:
                print("Step " + str(generation) +
                      ", best score: " + str(np.max(self.fitnesss)))

        print(f'End evol progress, best score ever: {self.best_fitness}')
        return

    def init(self):
        """Create and evaluate first generation in the environment
        """
        if self.fitness_func is None:
            raise ValueError("Please set fitness function before init")
        num_sastify_circuit = 0
        while (num_sastify_circuit < self.metadata.num_circuit):
            circuit = random_circuit.generate_with_pool(
                self.metadata.num_qubits, self.metadata.depth)
            if sastify_circuit(circuit):
                num_sastify_circuit += 1
                self.circuits.append(circuit)
        #         self.fitnesss.append(self.fitness_func(circuit))
        # self.circuitss.append(self.circuits)
        # self.metadata.fitnessss.append(self.fitnesss)
        # self.best_circuit = self.circuits[np.argmax(self.fitnesss)]
        # self.best_fitness = np.max(self.fitnesss)
        return

    def set_num_generation(self, num_generation: int) -> None:
        """Set new value for number of generation

        Args:
            _num_generation (_type_): _description_
        """
        self.metadata.num_generation = num_generation
        return

    def draw(self, file_name: str = ''):
        """Save all circuit from all generation at a specific path

        Args:
            file_name (str, optional): Path. Defaults to ''.
        """
        generation = 0
        for circuits in self.circuitss:
            generation += 1
            index = 0
            for circuit in circuits:
                path = f'{file_name}/{generation}'
                index += 1
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)
                circuit.draw('mpl').savefig(f'{path}/{index}.png')
        return

    def plot(self, metrics: str = ['best_fitness', 'average_fitness']):
        """Plot number of generation versus best score of each generation
        Example: ['best_fitness','average_fitness']
        """
        ticks_generation = list(range(1, self.metadata.current_generation, 1))
        for metric in metrics:
            if metric == 'best_fitness':
                plt.plot(ticks_generation,
                         np.max(np.array(self.metadata.fitnessss), axis=1), label=metric)
            if metric == 'average_fitness':
                plt.plot(ticks_generation,
                         np.mean(np.array(self.metadata.fitnessss), axis=1), label=metric)
        plt.legend()
        plt.xlabel('No. generation')
        plt.show()
        return

    @staticmethod
    def load(file_name: str, fitness_func: types.FunctionType):
        file = pathlib.Path(file_name)
        if file.is_dir():
            funcs = json.load(open(os.path.join(file_name, 'funcs.json')))
            metadata = json.load(
                open(os.path.join(file_name, 'metadata.json')))
            env = EEnvironment(
                metadata=metadata,
                crossover_func=getattr(crossover, funcs['crossover_func']),
                mutate_func=getattr(mutate, funcs['mutate_func']),
                selection_func=getattr(selection, funcs['selection_func']),
                threshold_func=getattr(threshold, funcs['threshold_func'])
            )
            env.circuitss = [[0 for _ in range(env.metadata.num_circuit)] for _ in range(
                env.metadata.current_generation)]
            for i in range(0, env.metadata.current_generation):
                for j in range(env.metadata.num_circuit):
                    env.circuitss[i][j] = utilities.load_circuit(os.path.join(file_name, f'circuit_{i + 1}_{j}'))
            env.set_fitness_func(fitness_func)
            env.set_circuits(env.circuitss[-1])
            env.best_circuit = utilities.load_circuit((os.path.join(file_name, 'best_circuit')))
        else:
            raise TypeError("Please input a path to json file or qsp folder")
        return env

    def save(self, file_name: str = ''):
        """Save as envobj file at a specific path

        Args:
            file_name (str): Path
        """
        if not os.path.exists(file_name):
            os.mkdir(file_name)

        funcs = {
            'fitness_func': self.fitness_func.__name__,
            'crossover_func': self.crossover_func.__name__,
            'mutate_func': self.mutate_func.__name__,
            'selection_func': self.selection_func.__name__,
            'threshold_func': self.threshold_func.__name__
        }
        with open(f"{os.path.join(file_name, 'metadata')}.json", "w") as file:
            json.dump(vars(self.metadata), file)
        with open(f"{os.path.join(file_name, 'funcs')}.json", "w") as file:
            json.dump(funcs, file)
        for i in range(0, self.metadata.current_generation):
            for j in range(self.metadata.num_circuit):
                utilities.save_circuit(
                    qc=self.circuitss[i][j],
                    file_name=os.path.join(file_name, f'circuit_{i + 1}_{j}')
                )
        utilities.save_circuit(
            qc=self.best_circuit,
            file_name=os.path.join(file_name, f'best_circuit')
        )
        return
