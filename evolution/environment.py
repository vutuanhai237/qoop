import types
import typing
import random
import datetime
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from .ecircuit import ECircuit
from .selection import sastify_circuit
from ..core import random_circuit
from ..backend import utilities
def extract_circuit(circuits: typing.List[ECircuit]):
    extracted_circuits = []
    for circuit in circuits:
        extracted_circuits.append(circuit.qc)
    return extracted_circuits

def bypass_compile(circuit: ECircuit):
    circuit.compile()
    return circuit
def multiple_compile(circuits: typing.List[ECircuit]):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(bypass_compile, circuits)
        return results
class EEnvironment():
    """Saved information for evolution process
    """
    def __init__(self, file_name: str):
        """_summary_

        Args:
            file_name (str): _description_
        """
        file = open(file_name, 'rb')
        data = pickle.load(file)
        self.__init__(
            data.params,
            data.fitness_func,
            data.crossover_func,
            data.mutate_func,
            data.selection_func,
            data.pool, data.file_name)
        file.close()
        return
    def __init__(self, params: typing.Union[typing.Dict, str],
                 fitness_func: types.FunctionType = None,
                 crossover_func: types.FunctionType = None,
                 mutate_func: types.FunctionType = None,
                 selection_func: types.FunctionType = None,
                 pool = None, file_name: str = '') -> None:
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
        if isinstance(params, str):
            file = open(params, 'rb')
            data = pickle.load(file)
            params = data.params
            self.params = data.params
            self.fitness_func = data.fitness_func
            self.crossover_func = data.crossover_func
            self.mutate_func = data.mutate_func
            self.selection_func = data.selection_func
            self.pool = data.pool
            self.file_name = data.file_name
            self.best_candidate = data.best_candidate
            self.current_generation = data.current_generation
            self.population = data.population
            self.populations = data.populations
            self.best_score_progress = data.best_score_progress
            self.scores_in_loop = data.scores_in_loop
        else:
            self.params = params
            self.fitness_func = fitness_func
            self.crossover_func = crossover_func
            self.mutate_func = mutate_func
            self.selection_func = selection_func
            self.pool = pool
            self.file_name = file_name
            self.best_candidate = None
            self.current_generation = 0
            self.population: typing.List[ECircuit] = []
            self.populations = []
            self.best_score_progress = []
            self.scores_in_loop = []
        self.depth = params['depth']
        self.num_circuit = params['num_circuit']  # Must mod 8 = 0
        self.num_generation = params['num_generation']
        self.num_qubits = params['num_qubits']
        self.prob_mutate = params['prob_mutate']
        self.threshold = params['threshold']
        self.predictor = None
        return
    def calculate_strength_point(self):
        inverse_fitnesss = [1- circuit.fitness for circuit in self.population]
        mean_inverse_fitnesss = np.mean(inverse_fitnesss)
        std_inverse_fitnesss = np.std(inverse_fitnesss)
        strength_points = [(1 - circuit.fitness - mean_inverse_fitnesss)/std_inverse_fitnesss for circuit in self.population]
        scaled_strength_points = utilities.softmax(strength_points, self.depth)
        for i, circuit in enumerate(self.population):
            circuit.strength_point = scaled_strength_points[i]
        return
    def set_predictor(self, predictor: Predictor):
        self.predictor = predictor
        return
    def evol(self, verbose: int = 1):
        # Pre-procssing
        if verbose == 1:
            bar = utilities.ProgressBar(
                max_value=self.num_generation, disable=False)
        
            
        if self.current_generation == 0:
            print("Initialize population ...")
            self.init()
            print("Start evol progress ...")
        elif self.current_generation == self.num_generation:
            return
        else:
            print(f"Continute evol progress at generation {self.current_generation} ...")
        for generation in range(self.current_generation, self.num_generation):
            #####################
            ##### Pre-process ###
            #####################
            print(f"Evol at generation {generation}")
            self.current_generation += 1
            self.scores_in_loop = []
            new_population = []
            offsprings = []
            #####################
            ##### Selection #####
            #####################
            if mode == NORMAL_MODE:
                self.population = multiple_compile(self.population)

            if mode == PREDICT_MODE:
                if self.predictor is None:
                    assert "Predictor is none, please set it by using set_predictor() method"
                return
            self.population = self.selection_func(self.population)
            #####################
            ##### Cross-over ####
            #####################
            for i in range(0, int(self.num_circuit/2), 2):
                offspring1, offspring2 = self.crossover_func(
                    self.population[i], self.population[i+1])
                offsprings.extend([offspring1, offspring2])
                new_population.extend([self.population[i], self.population[i + 1], ])
            for offspring in offsprings:
                print('k', offspring.fitness)
                new_population.append(offspring)
                self.scores_in_loop.append(offspring.fitness)
            self.population = new_population
            self.populations.append(self.population)
            ####################
            ##### Mutation #####
            ####################
            for i in range(0, len(self.population)):
                if random.random() < self.prob_mutate:
                    self.population[i] = self.mutate_func(self.population[i], self.pool)
                    
            #####################
            ##### Post-process ###
            #####################
            best_score = np.min(self.scores_in_loop)
            best_index = np.argmin(self.scores_in_loop)
            if self.best_candidate.fitness > self.population[best_index].fitness:
                self.best_candidate = self.population[best_index]
            self.best_score_progress.append(best_score)
            self.save(self.file_name + f'ga_{self.num_qubits}qubits_{self.fitness_func.__name__}_{datetime.datetime.now().strftime("%Y-%m-%d")}.envobj')
            if verbose == 1:
                bar.update(1)
            if verbose == 2 and generation % 5 == 0:
                print("Step " + str(generation) + ": " + str(best_score))
            if self.threshold(best_score):
                break
        print(f'End evol progress, best score ever: {best_score}')
        return

    def init(self, mode: str):
        """Create and evaluate first generation in the environment
        """
        self.population = []
        num_sastify_circuit = 0
        while(num_sastify_circuit <= self.num_circuit):
            random_circuit = qtm.random_circuit.generate_with_pool(
                self.num_qubits, self.depth, self.pool)
            if sastify_circuit(random_circuit):
                num_sastify_circuit += 1
                circuit = ECircuit(
                    random_circuit,
                    self.fitness_func)
                self.population.append(circuit)
        self.best_candidate = self.population[0]
        return
    
    def set_num_generation(self, _num_generation) -> None:
        """Set new value for number of generation

        Args:
            _num_generation (_type_): _description_
        """
        self.num_generation = _num_generation
        return
    def draw(self, file_name: str = ''):
        """Save all circuit from all generation at a specific path

        Args:
            file_name (str, optional): Path. Defaults to ''.
        """
        generation = 0
        for population in self.populations:
            generation += 1
            index = 0
            for circuit in population:
                path = f'{file_name}/{generation}'
                index += 1
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)
                circuit.qc.draw('mpl').savefig(f'{path}/{index}.png')
        return
    def plot(self, metrics: str):
        """Plot number of generation versus best score of each generation
        Example: ['best_fitness','average_fitness']
        """
        for metric in metrics:
            if metric == 'best_fitness':
                plt.plot(list(range(1, self.current_generation + 1)), self.best_score_progress, label = metric)
            if metric == 'average_fitness':
                average_fitness = []
                for generation in self.populations:
                    fitnesss = []
                    for circuit in generation:
                        fitnesss.append(circuit.fitness)
                    average_fitness.append(np.mean(fitnesss))
                plt.plot(list(range(1, self.current_generation + 1)), average_fitness, label = metric)
        plt.legend()
        plt.xlabel('No. generation')
        plt.show()
        return
    def save(self, file_name: str = ''):
        """Save as envobj file at a specific path

        Args:
            file_name (str): Path
        """
        file = open(file_name, 'wb')
        pickle.dump(self, file)
        file.close()
        return

    