from .codes import CombinatorialCode
import numpy as np
example_dictionary = {
    "eyes": [[], [1], [2], [2, 3], [1, 3]],
    "open not closed": [[1, 2, 3], [1, 2, 6], [1, 5, 6], [4, 5, 6], [3, 4, 5], [2, 3, 4], [1, 2], [1, 6], [5, 6], [4, 5], [3, 4], [2, 3], []],
    "closed not open": [[2, 3, 4, 5], [1, 2, 4], [1, 3, 5], [1, 4, 5], [1, 4], [1, 5], [2, 4], [3, 5], [4, 5], [4], [5], []],
    "example by Milo": [
        [], [0], [1], [7], [9], [10], [12], [13], [14], [0, 1], [0, 3], [0, 4], [1, 8], [2, 7], [3, 10], [4, 12], [5, 14],
        [8, 12], [9, 11], [10, 12], [0, 2, 11], [0, 6, 13], [1, 2, 4], [1, 9, 10], [2, 11, 14], [3, 9, 11], [4, 6, 9],
        [4, 7, 10], [4, 9, 10], [7, 11, 12], [10, 11, 14], [10, 12, 13], [0, 2, 6, 12], [2, 3, 5, 10], [0, 2, 3, 4, 8, 11],
        [4, 7, 8, 11, 12, 14]
    ],
    "random example 32":[[], [6, 18, 26, 30], [8, 9, 11, 16, 18], [0, 5, 13, 17, 19, 26], [2, 19, 26, 27, 28, 30], [4, 7, 9, 18, 19, 25], [7, 9, 12, 18, 21, 29], [8, 13, 14, 21, 23, 29], [9, 10, 12, 15, 20, 31], [7, 15, 19, 22, 24, 29, 30], [0, 6, 8, 11, 12, 20, 22, 31], [1, 3, 5, 10, 11, 14, 23, 30], [6, 10, 12, 16, 18, 19, 28, 30], [0, 6, 8, 17, 22, 24, 26, 29, 31], [0, 12, 13, 14, 16, 18, 22, 23, 27], [2, 6, 8, 9, 10, 15, 22, 28, 30], [2, 6, 10, 16, 18, 19, 21, 22, 28], [5, 7, 10, 13, 15, 16, 17, 24, 30], [9, 12, 18, 22, 23, 27, 28, 29, 30], [0, 2, 3, 4, 5, 6, 12, 19, 25, 26], [0, 2, 6, 8, 10, 17, 18, 20, 25, 28], [1, 3, 8, 12, 13, 14, 16, 21, 28, 29], [3, 5, 7, 8, 9, 11, 13, 16, 24, 25], [0, 1, 6, 7, 8, 9, 10, 12, 13, 25, 27], [0, 2, 3, 7, 12, 14, 15, 16, 24, 26, 27], [0, 4, 5, 7, 9, 13, 15, 20, 22, 25, 28], [1, 8, 10, 13, 15, 18, 19, 21, 24, 26, 30], [3, 5, 8, 10, 12, 14, 17, 18, 21, 23, 26], [0, 1, 4, 7, 9, 11, 14, 16, 22, 25, 26, 31], [0, 5, 6, 8, 15, 20, 23, 27, 28, 29, 30, 31], [1, 5, 8, 11, 15, 16, 18, 20, 21, 24, 27, 29, 31], [2, 3, 4, 5, 7, 9, 10, 13, 21, 22, 27, 28, 29], [5, 7, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 22]]
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
    return CombinatorialCode(R,"boolean_matrix")
