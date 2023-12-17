import numpy as np

def similarity(float1: float, float2: float, decay_rate=0.2) -> float:
    """
    From ChatGPT
    
    Calculate the similarity between two floating-point numbers.
    
    The similarity is defined using an exponential decay function, 
    where 1 means the numbers are equal, and 0 represents maximum dissimilarity.

    Args:
    - float1 (float): First floating-point number.
    - float2 (float): Second floating-point number.
    - decay_rate (float): Rate at which similarity decays. Higher values mean faster decay.

    Returns:
    - float: The similarity score in the range [0, 1].
    """
    difference = abs(float1 - float2)
    similarity = np.exp(-decay_rate * difference)
    return similarity