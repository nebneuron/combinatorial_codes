from typing import List
from numba.typed import List as NumbaList
from itertools import combinations
import numpy as np 
from numba import jit,typed, types
import copy 
possible_types={8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
def set_word_type(n_bits:int):
    """ Tthis function sets the type of the words in the code to be of n_bits
    Usage: set_word_type(n_bits)
    """
    if n_bits not in possible_types:
        raise ValueError("The number of bits is not supported")
    print(f"Setting the word type to {n_bits} bits")
    global WORD_TYPE
    WORD_TYPE=possible_types[n_bits]

set_word_type(16)

def empty_list(dtype=WORD_TYPE):
    return np.zeros(0,dtype=dtype)
def bit_order(a: int, b: int) -> bool:
    # Returns True if every bit in a is <= every bit in b (i.e., if a's 1s are a subset of b's 1s)
    return (a & ~b) == 0


# array_of_vectors = [[],[1], [9], [1,9],[-1,1, 22, 3, 4], [5, 6, 7, 8,  10],[1], [5,6],[7,8,10],[10],[1],[],[-1,1, 22, 3, 4]]

def boolean_matrix_to_array_of_words(B:np.ndarray, dtype=WORD_TYPE ) -> np.ndarray:
    """Here B is a numpy binary matrix. The function returns an array of lists of integers of type dtype
    Usage: w=boolean_matrix_to_array_of_words(B)
    """
    n,n_bits=B.shape
    if n_bits>16:
        raise ValueError("The number of columns in the boolean matrix is too large")
    if n==0:
        return np.zeros(0,dtype=dtype)
    powers_of_2 = 2**np.arange(n_bits, dtype=dtype)
    return (B @ powers_of_2).astype(dtype)


def array_of_words_to_boolean_matrix(words:np.ndarray, n_bits:int) -> np.ndarray:
    """Here words is an array of integers. The function returns a numpy binary matrix
    Usage: B=array_of_words_to_boolean_matrix(words, n_bits)
    """
    n=words.shape[0]
    if n==0:
        return np.zeros((0,n_bits),dtype=bool)
    return ((words.reshape(-1,1) >> np.arange(n_bits)) & 1).astype(bool)

def array_of_words_to_vectors_of_integers(words:np.ndarray,  n_bits:int) -> List[List[int]]:
    """Here words is an array of integers. The function returns a list of lists of integers
    Usage: v=array_of_words_to_vectors_of_integers(words)
    """
    n=words.shape[0]
    if n==0:
        return []
    B=array_of_words_to_boolean_matrix(words, n_bits)
    return [[int(y) for y in list(np.where(B[i,:])[0])] for i in range(n)]

@jit(nopython=True) # here we optimize for speed, as this function is a heavy lifter
def inclusion_relation(a: np.ndarray , b : np.ndarray) -> np.ndarray:
    """This evaluates the inclusion relation between two sets of words: a and b
    Args:
        a (np.ndarray): an array of n words a[i] of type WORD_TYPE
        b (np.ndarray): an array of m words b[i] of type WORD_TYPE
    Returns:
        np.ndarray: a matrix of size n x m of booleans. The entry (i,j) is True if a[i] is a subset of b[j]
    usage: R=inclusion_relation(a,b)
    """
    n=a.shape[0]
    m=b.shape[0]
    result = np.zeros((n, m), dtype=np.bool_) 
    for i in range(n):
        for j in range(m):
            result[i,j]=( (a[i] & ~b[j]) == 0 )
    return result

@jit(nopython=True) # here we optimize for speed, as this function is a heavy lifter
def x_is_a_subset_of_any_in_List(x: WORD_TYPE , L : NumbaList) -> bool:
    """This evaluates the inclusion relation between two sets of words: a and b
    Args:
        x : WORD_TYPE
        b (np.ndarray): an array of m words b[i] of type WORD_TYPE
    Returns:
        np.ndarray: a matrix of size n x m of booleans. The entry (i,j) is True if a[i] is a subset of b[j]
    usage: logical=x_is_a_subset_of_any_in_List(x,L)
    """
    for b in L:
            if ( type(b)(x) & ~( b)) == 0:
                return True
    return False



def sizes_of_words(words:np.ndarray) -> np.ndarray:
    """Here words is an array of integers. The function returns an array of integers
    Usage: s=sizes_of_words(words)
    """
    return np.array([ x.bit_count()  for i,x in enumerate(words)],dtype=WORD_TYPE)

def convert_to_boolean_matrix(array_of_vectors):
    """ The input array_of_vectors is an array of vectors that represent neurons in each codeword.
        The neuron numbers are assumed to be integers.
        B=convert_to_boolean_matrix(array_of_vectors)
    """
    if len(array_of_vectors)==0:
        return np.zeros((0,0),dtype=bool)
    # first we figure out the possible vertices
    vertex_set=set()
    for v in array_of_vectors: 
        vertex_set=vertex_set.union(v)
    vertex_set=sorted(list(vertex_set))
    n=len(vertex_set)
    # now we introduce a dictionary that maps our vertices into non-negative numbers in the range(n)
    d={vertex_set[i] :i for i in range(n) }
    B=np.zeros((len(array_of_vectors),n),dtype=bool)
    for (i,v) in enumerate(array_of_vectors):
        B[i, [d[x] for x in array_of_vectors[i]] ]=True
    return B



def find_maximal_words(words:np.ndarray,sizes: np.ndarray) -> np.ndarray:
    """Here words is an array of integers. The function returns an array of integers
    Usage: s=sizes_of_words(words)
    """
    n=words.shape[0]
    if n!=sizes.shape[0]:
        raise ValueError("The number of words and sizes do not match")
    if n==0:
        return np.zeros(0,dtype=WORD_TYPE)
    unique_sizes, indices = np.unique(sizes, return_inverse=True)
    indices_by_size=dict({ u:  np.where(indices==i)[0] for (i,u) in enumerate(unique_sizes)})
    max_words=NumbaList(words[indices_by_size[ unique_sizes.max()]])
    for u in range(len(unique_sizes)-1,-1,-1):
        for w in words[indices_by_size[unique_sizes[u]]]:
             if not x_is_a_subset_of_any_in_List(w, max_words):
                max_words.append(w)
    return np.array(max_words,dtype=WORD_TYPE)


class CombinatorialCode:
    def __init__(self, array_of_vectors: List[List[int]]):
        if not array_of_vectors:
            # No words were passed
            self.words = np.zeros((0,0),dtype=bool)
            self.sizes = []
            self.max_size = 0
            self.min_size = 0
            self.n_words = 0
            self.n_bits=0
            self.maximal_words_idx = []
        else:
            B=convert_to_boolean_matrix(array_of_vectors)
            words=np.unique(boolean_matrix_to_array_of_words(B))
            sizes=sizes_of_words(words)
            self.words=words
            self.sizes=sizes
            self.n_bits=B.shape[1]
            self.n_words=words.shape[0]
            unique_sizes, indices = np.unique(sizes, return_inverse=True)
            self.unique_sizes = unique_sizes
            self.indices_by_size=dict({ u:  np.where(indices==i)[0] for (i,u) in enumerate(unique_sizes)})
            self.min_size =unique_sizes.min()
            self.max_size =unique_sizes.max()
            self.maximal_words=  find_maximal_words(words,sizes)
    def has_empty_set(self):
        return bool(self.min_size==0)
    def has_full_set(self):
        return bool(self.max_size==self.n_bits)
    def __repr__(self):
        header=f"CombinatorialCode with {self.n_words} words of {self.n_bits} bits\n"
        if self.n_words==0:
            return header
        else:
            nonmaximal_words = self.words[~np.isin(self.words, self.maximal_words)]
            maximal_words_line=f"Maximal words: {array_of_words_to_vectors_of_integers(self.maximal_words,self.n_bits)}\n"
            nonmaximal_words_line=f"Non-maximal words: {array_of_words_to_vectors_of_integers(nonmaximal_words,self.n_bits)}\n"
            return header+ maximal_words_line+nonmaximal_words_line  +"\n"
    def show(self):
        print(self.__repr__())