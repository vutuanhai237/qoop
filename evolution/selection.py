import numpy as np, random
from .utilities import sort_by_fitness
def sastify_circuit(qc):
    if len(qc.parameters) == 0:
        return False
    return True
    
def elitist_selection(objects, fitnesss, num_elitist = 0):
    if num_elitist == 0:
        num_elitist = int(len(objects)/2)
    sorted_object = sort_by_fitness(objects, fitnesss)
    return sorted_object[:num_elitist]



def tournament_selection(population, fitnesses):
    def find_tournament_size(population_size):
        a = round(int(np.sqrt(population_size)))
        if population_size % a == 0:
            return a
        for i in range(1, int(np.sqrt(population_size)) + 1):
            if population_size % (a - i) == 0:
                return a - i
            elif population_size % (a + i) == 0:
                return a + i
        return None
    pop_with_fitness = list(zip(population, fitnesses))
    winner = []
    # Calculate the tournament_size such as tournament_size is nearly equal to int(len(population)/tournament_size)
    tournament_size = find_tournament_size(len(population))
    for i in range(0, int(len(population)/tournament_size)):
        tournament = random.sample(pop_with_fitness, tournament_size)
        for t in tournament:
            pop_with_fitness.remove(t)
        winner.append(max(tournament, key=lambda x: x[1]))
    return winner


def roulette_wheel_selection(objects, fitnesss, num_selection=0):
    """Roulette wheel selection

    Args:
        - objects (list): List of objects
        - fitnesss (list): List of fitnesss
        - num_elitist (int, optional): Number of elitist. Defaults to 0.

    Returns:
        - selected_object (list): List of selected objects
    """
    if num_selection == 0:
        num_selection = int(len(objects)/2)
    selected_object = []
    fitnesss = np.array(fitnesss)
    fitnesss = fitnesss/np.sum(fitnesss)
    print(fitnesss)
    for i in range(len(objects)-num_selection):
        object = (np.random.choice(objects, p=fitnesss))
        selected_object.append(object)
    return selected_object