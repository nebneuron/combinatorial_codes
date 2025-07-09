from typing import List
from numba.typed import List as NumbaList
from numba.np.numpy_support import from_dtype
from itertools import combinations
import numpy as np 
from numba import jit, njit, typed, types
from numba.typed import Dict # we need this to define a typed dictionary 
import gudhi
from .tda import homology_is_trivial
from .utils import * 


""" 
Example: 
C=example_code( "example by Milo")
violators=C.simplicial_violators()
array_representation = array_of_words_to_vectors_of_integers(violators, C.n_bits)
print('simplicial violaters: ', array_representation)
obstruction_information=C.Obstructions()
"""



# array_of_vectors = [[],[1], [9], [1,9],[-1,1, 22, 3, 4], [5, 6, 7, 8,  10],[1], [5,6],[7,8,10],[10],[1],[],[-1,1, 22, 3, 4]]

def boolean_matrix_to_array_of_words(B:np.ndarray, dtype ) -> np.ndarray:
    """Here B is a numpy binary matrix. The function returns an array of lists of integers of type dtype
    Usage: w=boolean_matrix_to_array_of_words(B)
    """
    n,n_bits=B.shape
    if n_bits>MAX_NUMBER_OF_BITS:
        raise ValueError("The number of columns in the boolean matrix is too large")
    if n==0:
        return np.zeros(0,dtype=dtype)
    powers_of_2 = 1<< np.arange(n_bits, dtype=dtype) # 2**np.arange(n_bits, dtype=dtype)
    return (B @ powers_of_2).astype(dtype)

@njit
def array_of_words_to_boolean_matrix(words:np.ndarray, n_bits:int) -> np.ndarray:
    """Here words is an array of integers. The function returns a numpy binary matrix
    Usage: B=array_of_words_to_boolean_matrix(words, n_bits)
    """
    n=len(words)
    if n==0:
        return np.zeros((0,n_bits),dtype=np.bool_)
    return ((words.reshape(-1,1) >> np.arange(n_bits)) & 1).astype(np.bool_)


def array_of_words_to_vectors_of_integers(words:np.ndarray,  n_bits:int, translation_dict:Dict=None) -> List[List[int]]:
    """Here words is an array of integers. The function returns a list of lists of integers
    Usage: v=array_of_words_to_vectors_of_integers(words)
    """
    n=len(words)
    if n==0:
        return []
    B=array_of_words_to_boolean_matrix(words, n_bits)
    if translation_dict is None:
        result= [[int(y) for y in list(np.where(B[i,:])[0])] for i in range(n)]
    else:
         result=  [[translation_dict[int(y)] for y in list(np.where(B[i,:])[0])] for i in range(n)]
    return sorted(result, key=lambda x: (len(x), x)) # sort by length and then lexicographically

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


def convert_to_array_of_words(array_of_vectors):
    """ 
    converted_words, inverse_d=convert_to_array_of_words(array_of_vectors)
    The input array_of_vectors is an array of vectors that represent neurons in each codeword.
    The neuron numbers are assumed to be integers. These integers do not need to be consecutive, or even positive.        
    The function returns a tuple of two elements:
    - converted_words: an array of integers that represent the codewords
    - inverse_d: a dictionary that maps our vertices into non-negative numbers in the range(n)
    """
    n_words=len(array_of_vectors)
    # first we figure out the possible vertices
    vertex_set=set()
    for v in array_of_vectors: 
        vertex_set=vertex_set.union(v)
    vertex_set=sorted(list(vertex_set))
    n_bits=len(vertex_set)
    # precompute the shift table
    shift_table=np.array( [1<<i for i in range(n_bits)]  ,dtype=WORD_TYPE)
    if n_bits>MAX_NUMBER_OF_BITS:
        raise ValueError("The number of bits ={n_bits} is too large. Currently we do not allow more than {MAX_NUMBER_OF_BITS}. ")
    # d is  dictionary that maps our vertices into non-negative numbers in the range(n)
    d={vertex_set[i] :i for i in range(n_bits) }
    inverse_d={d[k]:k for k in d}
    converted_words=np.zeros(n_words,dtype=WORD_TYPE)
    for (i,v) in enumerate(array_of_vectors):
        x=np.array([d[k] for k in v],dtype=np.int64)
        converted_words[i]=shift_table[x].sum()
    return  converted_words, inverse_d










class CombinatorialCode:
    """This class is a container for a combinatorial code. 
    It is designed to be initialized in the following way:
    C=CombinatorialCode(array_of_vectors)
    where array_of_vectors is a list of lists of integers that represent the codewords.
    Note that incremental addition of codewords is _not_  implemented, 
    and would be very cumbersome and inefficient.
    The class has the following attributes:
    - words: an array of integers that represent the codewords
    - dtype: the type of the words
    - sizes: an array of integers that represent the sizes of the codewords
    - n_words: the number of words in the code
    - n_bits: the number of bits in each word
    - maximal_words : an array of integers that represent the maximal codewords
    The class has the following methods:
    - has_empty_set(): returns True if the code has an empty set
    - has_full_set(): returns True if the code has a full set
    - __repr__(): returns a string representation of the code
    - show(): prints the string representation
"""
    def __init__(self, array_of_vectors: List[List[int]], method: str = "array_of_vectors"):
        possible_methods = ["array_of_vectors", "numpy codewords", "boolean_matrix"]
        if method not in possible_methods:
            raise ValueError(f"Invalid method: {method}. Possible methods are: {possible_methods}")
        if len(array_of_vectors)==0:
            # No words were passed
            self.words = np.zeros(0,dtype=np.uint8)
            self.dtype=np.uint8
            self.sizes = []
            self.max_size = 0
            self.min_size = 0
            self.n_words = 0
            self.n_bits=0
            self.maximal_words= self.words
        else:
            self.dtype= WORD_TYPE 
            if method == "array_of_vectors":
                converted_words, translation_dict=convert_to_array_of_words(array_of_vectors)
                n_bits=max(list(translation_dict.keys()))+1
                words=np.unique(converted_words)
            elif method=="boolean_matrix":
                B=array_of_vectors
                n_bits=B.shape[1]
                words=np.unique(boolean_matrix_to_array_of_words(B,self.dtype))
                translation_dict={i:i for i in range(n_bits)}
            else: 
                raise ValueError(f"Invalid method: {method}. The other methods are not implemented yet.")
            # now we can set the attributes
            self.n_bits=n_bits
            sizes=np.array([ x.bit_count()  for x in words]) 
            self.translation_dict=translation_dict
            sizes=np.array([ x.bit_count()  for x in words])     
            self.unique_sizes, indices = np.unique(sizes, return_inverse=True)
            indices_by_size=Dict.empty(key_type=types.int64, value_type=types.Array(types.int64, 1, 'C'))
            for (i,u) in enumerate(self.unique_sizes):
                indices_by_size[int(u)]=   np.where(indices == i)[0]
            self.indices_by_size=indices_by_size
            self.words=words
            self.sizes=sizes
            self.n_words=words.shape[0]
            self.min_size, self.max_size = self.unique_sizes.min(), self.unique_sizes.max()
            self.maximal_words= find_maximal_words(words, self.unique_sizes, indices_by_size,self.dtype)
    def has_empty_set(self):
        return bool(self.min_size==0)
    def has_full_set(self):
        return bool(self.max_size==self.n_bits)
    def simplicial_violators(self, enforce_maximal_word_limit: bool=True):
        return simplicial_violators_from_words(self.words, self.maximal_words, enforce_maximal_word_limit)
    def Obstructions(self):
        return Obstructions(self)
    def __repr__(self):
        header=f"CombinatorialCode with {self.n_words} words of {self.n_bits} bits\n"
        if self.n_words==0:
            return header
        else:
            nonmaximal_words = self.words[~np.isin(self.words, self.maximal_words)]
            maximal_words_line=f"Maximal words: {array_of_words_to_vectors_of_integers(self.maximal_words,self.n_bits,self.translation_dict)}\n"
            nonmaximal_words_line=f"Non-maximal words: {array_of_words_to_vectors_of_integers(nonmaximal_words,self.n_bits,self.translation_dict)}\n"
            return header+ maximal_words_line+nonmaximal_words_line  +"\n"
    def show(self):
        print(self.__repr__())






def Obstructions(C: CombinatorialCode):
    """ is_maximal_intersection_complete, N_obstructions=Obstructions(C)
    Args:
        C (CombinatorialCode): a combinatorial code
    Returns:
       is_maximal_intersection_complete: a boolean that is True if the code is maximal intersection complete
         N_obstructions: the number of obstructions to being a good code in the sense of the paper
    """

    if not C.has_empty_set():
        raise ValueError("The code does not have an empty set. This function can only handle codes with empty sets.")
    
    if C.has_full_set():
         return (True, 0)
    else: 
        simplicial_violators=simplicial_violators_from_words(C.words, C.maximal_words)
        is_maximal_intersection_complete=(len(simplicial_violators)==0)
        if is_maximal_intersection_complete:
            return (True, 0)
    # Now we compute the local obstructions for each simplicial violator
    N_violators=len(simplicial_violators)
    has_obstructions=np.empty(N_violators,dtype=np.bool_)
    for i,v in enumerate(simplicial_violators):
        facets=link_facets(v, C.maximal_words)
        arr=array_of_words_to_vectors_of_integers(facets,C.n_bits)
        has_obstructions[i]= (not homology_is_trivial(arr))
    N_obstructions=has_obstructions.sum()
    return is_maximal_intersection_complete, int(N_obstructions)



