

from .ecircuit import ECircuit
from .selection import sastify_circuit
from ..evolution import crossover, mutate, selection, threshold, generator
from ..core import random_circuit
from ..backend import utilities
from .environment_parent import Metadata
import types
import typing
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


# def multiple_compile(circuits: typing.List[ECircuit]):
#     executor = concurrent.futures.ProcessPoolExecutor()
#     results = executor.map(bypass_compile, circuits)
#     return results

def multiple_compile(func, params):
    k = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for number in (executor.map(func, params)):
            k.append((number))
    return k

class EEnvironment():
    """Saved information for evolution process
    """

    def __init__(self, metadata: Metadata | dict,
                 fitness_func: typing.List[types.FunctionType] = [],
                 generator_func: types.FunctionType = None,
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
        if callable(fitness_func):
            self.fitness_func = fitness_func
        elif len(fitness_func) == 1:
            self.fitness_func = fitness_func
        elif len(fitness_func) == 2:
            self.fitness_func = fitness_func[0]
            self.fitness_full_func = fitness_func[1]
        else:
            self.fitness_func = None
        self.generator_func = generator_func
        self.crossover_func = crossover_func
        self.mutate_func = mutate_func
        self.selection_func = selection_func
        self.threshold_func = threshold_func
        if isinstance(metadata, Metadata):
            self.metadata = metadata
        # Eliminate this case because it will be many type of env_metadata
        # elif isinstance(metadata, dict):
        #     self.metadata = Metadata(
        #         num_qubits=metadata.get('num_qubits', 0),
        #         depth=metadata.get('depth', 0),
        #         num_circuit=metadata.get('num_circuit', 0),
        #         num_generation=metadata.get('num_generation', 0),
        #         current_generation=metadata.get('current_generation', 0),
        #         fitnessss=metadata.get('fitnessss', []),
        #         best_fitnesss=metadata.get('best_fitnesss', []),
        #         prob_mutate=metadata.get('prob_mutate', []),
        #     )
        self.fitnesss: list = []
        self.circuits: typing.List[ECircuit] = []
        self.circuitss: typing.List[typing.List[ECircuit]] = []
        self.best_circuit = None
        self.best_circuits: typing.List[ECircuit] = []
        self.best_fitness = 0
        self.file_name = None
        return

    def set_filename(self, file_name: str):
        self.file_name = file_name
        return 
    
    def set_fitness_func(self, fitness_func):
        self.fitness_func = fitness_func
        return

    def set_circuitss(self, circuitss):
        self.circuitss: typing.List[typing.List[qiskit.QuantumCircuit]] = circuitss
        return

    def set_best_circuits(self, circuits):
        self.best_circuits: typing.List[qiskit.QuantumCircuit] = circuits
        return
    
    def set_circuits(self, circuits):
        self.circuits: typing.List[qiskit.QuantumCircuit] = circuits
        return

    def evol(self, verbose: int = 0, mode = 'parallel', auto_save: bool = True):
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
            if verbose == 1:
                bar.update(1)
            if verbose == 2 and generation % 5 == 0:
                print("Generation " + str(self.metadata.current_generation) +
                      ", best score: " + str(np.max(self.fitnesss)))
            print(f"Running at generation {self.metadata.current_generation}")
            
            #####################
            ######## Cost #######
            #####################
            # new_population = multiple_compile(new_population)
            if mode == 'parallel':
                self.fitnesss = []
                fitnesss_temp = multiple_compile(self.fitness_func, self.circuits)
                self.fitnesss.extend(fitnesss_temp)
            else:
                for i in range(len(self.circuits)):
                    self.fitnesss.append(self.fitness_func(self.circuits[i]))
            self.metadata.best_fitnesss.append(np.max(self.fitnesss))
            self.best_circuits.append(self.circuits[np.argmax(self.fitnesss)])
            if self.best_circuit is None:
                self.best_circuit = self.best_circuits[0]
            print(np.round(self.fitnesss, 4))
            self.metadata.fitnessss.append(self.fitnesss)
            if auto_save:
                self.save()
            #####################
            #### Threshold ######
            #####################
            if self.best_fitness < np.max(self.fitnesss):
                self.best_circuit = self.circuits[np.argmax(self.fitnesss)]
                self.best_fitness = np.max(self.fitnesss)
                if hasattr(self, 'fitness_full_func'):
                    full_best_fitness = self.fitness_full_func(self.best_circuit)
                    if self.threshold_func(full_best_fitness):
                        print(
                            f'End progress soon at generation {self.metadata.current_generation}, best score ever: {full_best_fitness}')
                        return self
                else:
                    if self.threshold_func(self.best_fitness):
                        print(
                            f'End progress soon at generation {self.metadata.current_generation}, best score ever: {self.best_fitness}')
                        return self

            #####################
            ##### Selection #####
            #####################
            self.circuits = self.selection_func(self.circuits, self.fitnesss)
            #####################
            ##### Cross-over ####
            #####################
            new_circuits: typing.List[ECircuit] = []
            for i in range(0, int(self.metadata.num_circuit/2), 2):
                offspring1, offspring2 = self.crossover_func(
                    self.circuits[i], self.circuits[i + 1])
                new_circuits.extend(
                    [self.circuits[i], self.circuits[i + 1], offspring1, offspring2])

            ####################
            ##### Mutation #####
            ####################
            for i in range(0, len(new_circuits)):
                new_circuits[i] = self.mutate_func(new_circuits[i])
                # normalize parameter of circuit
                new_circuits[i] = utilities.compose_circuit([new_circuits[i]])
            #####################
            ##### Post-process ##
            #####################
            self.circuits = new_circuits
            self.circuitss.append(new_circuits)
            if auto_save:
                self.save()      

        print(f'End evol progress, best score ever: {self.best_fitness}')
        return self

    def init(self):
        """Create and evaluate first generation in the environment
        """
        if self.fitness_func is None:
            raise ValueError("Please set fitness function before init")
        num_sastify_circuit = 0
        while (num_sastify_circuit < self.metadata.num_circuit):
            circuit = self.generator_func(self.metadata)
            if sastify_circuit(circuit):
                num_sastify_circuit += 1
                self.circuits.append(circuit)
        # self.circuitss.append(self.circuits)
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
        ticks_generation = list(range(1, self.metadata.current_generation + 1, 1))
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
            try:
                funcs = json.load(open(os.path.join(file_name, 'funcs.json')))
                metadata = json.load(
                    open(os.path.join(file_name, 'metadata.json')))
                env = EEnvironment(
                    metadata=metadata,
                    generator_func=getattr(generator, funcs['generator_func']),
                    crossover_func=getattr(crossover, funcs['crossover_func']),
                    mutate_func=getattr(mutate, funcs['mutate_func']),
                    selection_func=getattr(selection, funcs['selection_func']),
                    threshold_func=getattr(threshold, funcs['threshold_func'])
                )
                env.circuitss = [[0 for _ in range(env.metadata.num_circuit)] for _ in range(
                    env.metadata.current_generation)]
                env.best_circuits = [0]*(env.metadata.current_generation)
                for i in range(0, env.metadata.current_generation - 1):
                    env.best_circuits[i] = utilities.load_circuit(os.path.join(file_name, f'best_circuit_{i + 1}'))
                    for j in range(env.metadata.num_circuit):
                        env.circuitss[i][j] = utilities.load_circuit(os.path.join(file_name, f'circuit_{i + 1}_{j}'))
                env.set_fitness_func(fitness_func)
                env.set_circuits(env.circuitss[-1])
            except:
                return
            env.best_circuit = utilities.load_circuit((os.path.join(file_name, 'best_circuit')))

        else:
            raise TypeError("Please input a path to env folder")
        return env
    def draw(self, file_name: str = None):
        fig, ax = plt.subplots(len(self.circuitss),self.metadata.num_circuit)
        for i in range(0, len(self.circuitss)):
            for j in range(self.metadata.num_circuit):
                self.circuitss[i][j].draw('mpl', ax=ax[i][j])
        if file_name is not None:
            plt.savefig(f'{file_name}.png', dpi = 1000)
        plt.show()
        return  

    def draw_best_circuit(self, file_name: str = None):
        fig, ax = plt.subplots(1, len(self.circuitss))
        for i in range(0, len(self.best_circuits)):
            self.best_circuits[i].draw('mpl', ax=ax[i])
        if file_name is not None:
            plt.savefig(f'{file_name}.png', dpi = 1000)
        plt.show()
        return  
    
    def save(self, file_name: str = ''):
        """Save as envobj file at a specific path

        Args:
            file_name (str): Path
        """
        if self.file_name is not None:
            file_name = self.file_name
        else:
            file_name = f'{self.metadata.num_qubits}qubits_{self.fitness_func.__name__}_{datetime.datetime.now().strftime("%Y-%m-%d")}'
    
        if not os.path.exists(file_name):
            os.mkdir(file_name)

        funcs = {
            'generator_func': self.generator_func.__name__,
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
        print(f"Saving circuit ...")
        for i in range(0, len(self.circuitss)):
            for j in range(self.metadata.num_circuit):
                utilities.save_circuit(
                    qc=self.circuitss[i][j],
                    file_name=os.path.join(file_name, f'circuit_{i + 1}_{j}')
                )
        for i in range(0, len(self.best_circuits)):
            utilities.save_circuit(
                    qc=self.best_circuits[i],
                    file_name=os.path.join(file_name, f'best_circuit_{i + 1}')
                )
        if self.best_circuit is not None:
            utilities.save_circuit(
                qc=self.best_circuit,
                file_name=os.path.join(file_name, f'best_circuit')
            )
        return
