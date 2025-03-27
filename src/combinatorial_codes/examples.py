from combinatorial_codes import CombinatorialCode
import numpy as np
example_dictionary= {
    "eyes" : [[],[1],[2],[2,3],[1,3]],
    "open not closed" : [[1, 2, 3], [1, 2, 6], [1, 5, 6], [4, 5, 6], [3, 4, 5], [2, 3, 4], [1, 2], [1, 6], [5, 6], [4, 5], [3, 4], [2, 3], []],
    "closed not open":  [[2, 3, 4, 5], [1, 2, 4], [1, 3, 5], [1, 4, 5], [1, 4], [1, 5], [2, 4], [3, 5], [4, 5], [4], [5], []]
    }

def example_code(name: str):
    if name not in example_dictionary:
        print("Available names are the following:") 
        for n in example_dictionary.keys(): print(n)
        raise ValueError("Invalid name.") 
    return CombinatorialCode(example_dictionary[name])


def bernoulli_random_code(n_bits: int, Nwords: int, p: float):
    """
    C=bernoulli_random_code(n_bits, Nwords, p)
    
    Draw Nwords i.i.d. samples from the distribution on N-bit words given by N independent
    Bernoulli trials with success probability p.
    
    Parameters:
        n_bits (int): Number of bits per word.
        Nwords (int): Number of words (samples) to draw.
        p (float): Success probability for each Bernoulli trial.
    
    Returns:
        An instance of CombinatorialCode constructed with the generated boolean matrix.
    """
    # Generate an (Nwords x n_bits) array of uniform random numbers in [0, 1),
    # then transform it to have the same distribution as 1 - U.
    # Checking (1 - U) <= p produces True with probability p.
    R = (1 - np.random.rand(Nwords, n_bits)) <= p
    return CombinatorialCode([np.where(R[i])[0] for i in range(Nwords)])