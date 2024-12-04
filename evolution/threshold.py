
def synthesis_threshold(fitness_value):
    if fitness_value > 0.99:
        return True
    return False

def compilation_threshold(fitness_value):
    if fitness_value > 0.9:
        return True
    return False

def vqe_threshold(GA_eigenvalue):
    if abs(GA_eigenvalue - ene_vqe) < 0.01:
        return True
    return False