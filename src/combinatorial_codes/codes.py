from typing import List
from numba.typed import List as NumbaList
from numba.np.numpy_support import from_dtype
from itertools import combinations
import numpy as np 
from numba import jit, njit, typed, types
from numba.typed import Dict # we need this to define a typed dictionary 
import gudhi
# from tda import homology_is_trivial

# Define a Numba array type: 1D float64 array in C-contiguous layout.
array_type = types.Array(types.int64, 1, 'C')
MaximalWordLimit=20; # this is the maximal number of maximal words that we can handle
SizeType=np.uint8 # no more than size =64 is supported for computing nerves of maximal words
# possible_types={8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}

WORD_TYPE=np.uint32
WORD_TYPE_NUMBA = from_dtype(np.dtype(WORD_TYPE))
WORD_TYPE_numba=types.uint32

# Precompute the corresponding Numba types.
KEY_TYPE = from_dtype(np.dtype(SizeType))
VALUE_TYPE = types.Array(from_dtype(np.dtype(WORD_TYPE)), 1, 'C')



"""  the fillowing is not currently used, but it is a good idea to keep it for future reference
WORD_TYPE8=np.uint8
WORD_TYPE16=np.uint16
WORD_TYPE32=np.uint32
WORD_TYPE64=np.uint64
@njit
def set_word_type(n_bits:int):
    possible_types_keys=np.array([8,16,32,64],dtype=np.uint16)
    if n_bits > 64:
        raise ValueError(f"The number of bits larger than {64} is not supported")
    actual_n_bits=min([x for x in possible_types_keys if x>=n_bits])
    # print(f"Setting the word type to {actual_n_bits} bits")
    if actual_n_bits == 8:
        return WORD_TYPE8
    elif actual_n_bits == 16:
        return WORD_TYPE16
    elif actual_n_bits == 32:
        return WORD_TYPE32
    elif actual_n_bits == 64:
        return WORD_TYPE64
    else:
        raise ValueError("Unsupported bit width")
"""

# some combinatorial helper functions, implemented using numba
@njit
def comb(n, k):
    """Compute the binomial coefficient 'n choose k'."""
    if k > n:
        return 0
    res = 1
    for i in range(1, k + 1):
        res = res * (n - i + 1) // i
    return res

@njit
def count_bits(x):
    count = 0
    while x:
        count += 1
        x = x & (x - 1)
    return count




@njit
def custom_bit_length(x):
    """
    Compute the bit length of a positive integer x.
    Returns 0 if x is 0.
    """
    temp=x
    length = 0
    while temp:
        length += 1
        temp //= 2  # integer division
    return length


@jit(nopython=True)
def intersection_of_codewords_from_bits(x, a):
    """
    Given:
      - x: an  unsigned integer
      - a: a NumPy vector of numbers (with length >= the number of set bits in x)
    Returns a single numpy unsigned integer that represents the intersection of the codewords 
    in a that correspond to the bits set in x.
    """
    if x==0:
        raise ValueError("The input x must be a positive integer")
    dtype_x=type(x)
    dtype_a=type(a[0])
    encountered_set_bit=False
    for i in range(custom_bit_length(x)):
        # Check if the i-th bit is set.
        # Using bitwise shift: (1 << i) is computed as dtype_x(1) << dtype_x(i).
        if x & (dtype_x(1) << dtype_x(i)):
            if not encountered_set_bit:
                result=a[i]
                encountered_set_bit=True
            else:
                result=  result  &   a[i]
    return  dtype_a(result)














def empty_list(dtype):
    return np.zeros(0,dtype=dtype)

def bit_order(a: int, b: int) -> bool:
    # Returns True if every bit in a is <= every bit in b (i.e., if a's 1s are a subset of b's 1s)
    return (a & ~b) == 0


# array_of_vectors = [[],[1], [9], [1,9],[-1,1, 22, 3, 4], [5, 6, 7, 8,  10],[1], [5,6],[7,8,10],[10],[1],[],[-1,1, 22, 3, 4]]

@jit(nopython=True) 
def generate_binary_strings(n : int ) -> np.ndarray:
    """ Lattice=generate_binary_strings(n)
    Returns a boolean matrix of shape (2**n, n) 
    where each row is a different combination of True/False values,
    sampling the entire Boolean lattice.
    """
    num_strings = 1<< n # 2**n
    binary_strings = np.zeros((num_strings, n), dtype=np.bool_)
    for i in range(num_strings):
        for j in range(n):
            binary_strings[i, j] = (i >> j) & 1
    return binary_strings




def boolean_matrix_to_array_of_words(B:np.ndarray, dtype ) -> np.ndarray:
    """Here B is a numpy binary matrix. The function returns an array of lists of integers of type dtype
    Usage: w=boolean_matrix_to_array_of_words(B)
    """
    n,n_bits=B.shape
    if n_bits>16:
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


def array_of_words_to_vectors_of_integers(words:np.ndarray,  n_bits:int) -> List[List[int]]:
    """Here words is an array of integers. The function returns a list of lists of integers
    Usage: v=array_of_words_to_vectors_of_integers(words)
    """
    n=len(words)
    if n==0:
        return []
    B=array_of_words_to_boolean_matrix(words, n_bits)
    return [[int(y) for y in list(np.where(B[i,:])[0])] for i in range(n)]

@jit(nopython=True) # here we optimize for speed, as this function is a heavy lifter
def inclusion_relation(a: np.ndarray , b : np.ndarray) -> np.ndarray:
    """This evaluates the inclusion relation between two sets of words: a and b
    Args:
        a (np.ndarray): an array of n words a[i] of type dtype
        b (np.ndarray): an array of m words b[i] of type dtype
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

@njit
def link_facets(x: WORD_TYPE, maximal_words: np.ndarray) -> np.ndarray:
    """ facets=link_facets(x, maximal_words)
    Args:
        x (WORD_TYPE): a word
        maximal_words (numpy.ndarray): an array of maximal words
    Returns:
        facets: a list of the facets of the link_x(simplicial complex( maximal_words))
        Here each facet has the type of WORD_TYPE
    """
    m=len(maximal_words)
    x_in_facets=np.array([ ((x & f)==x) for f in maximal_words],dtype=np.bool_)
    relevant_maximal_words=maximal_words[x_in_facets]
    n_facets=relevant_maximal_words.shape[0]
    facets = np.empty( n_facets, dtype=WORD_TYPE)
    for i in range(n_facets):
        facets[i]=(~x) & relevant_maximal_words[i]
    return facets



@jit(nopython=True) # here we optimize for speed, as this function is a heavy lifter
def x_is_a_subset_of_any_in_List(x: type, L : NumbaList) -> bool:
    """This evaluates the inclusion relation between two sets of words: a and b
    Args:
        x : dtype
        b (np.ndarray): an array of m words b[i] of type dtype
    Returns:
        np.ndarray: a matrix of size n x m of booleans. The entry (i,j) is True if a[i] is a subset of b[j]
    usage: logical=x_is_a_subset_of_any_in_List(x,L)
    """
    for b in L:
            if ( type(b)(x) & ~( b)) == 0:
                return True
    return False





@jit(nopython=True) # here we optimize for speed, as this function is a heavy lifter
def x_is_a_superset_of_any_in_List(x: type, L : NumbaList) -> bool:
    """This evaluates the inclusion relation between two sets of words: a and b
    Args:
        x : dtype
        b (np.ndarray): an array of m words b[i] of type dtype
    Returns:
        np.ndarray: a matrix of size n x m of booleans. The entry (i,j) is True if a[i] is a subset of b[j]
    usage: logical=x_is_a_subset_of_any_in_List(x,L)
    """
    if len(L)==0:
        return False
    for b in L:
            if ( b & ~( type(b)(x))) == 0:
                return True
    return False










def sizes_of_words(words:np.ndarray) -> np.ndarray:
    """Here words is an array of integers. The function returns an array of integers
    Usage: s=sizes_of_words(words)
    """
    return np.array([ x.bit_count()  for i,x in enumerate(words)])




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


@jit(nopython=True) # here we optimize for speed, as this function is a heavy lifter
def find_maximal_words(words:np.ndarray,  unique_sizes: np.ndarray , indices_by_size : Dict, dtype) -> np.ndarray:
    """Here words is an array of integers. The function returns an array of integers
    Usage: s=sizes_of_words(words)
    """
    n=words.shape[0]
    if n==0:
        return np.zeros(0,dtype= dtype)
    max_size=  unique_sizes.max()
    max_words=NumbaList(words[indices_by_size[ max_size]])
    for u in range(len(unique_sizes)-1,-1,-1):
        for w in words[indices_by_size[unique_sizes[u]]]:
             if not x_is_a_subset_of_any_in_List(w, max_words):
                max_words.append(w)
    # finally we return the result as a numpy array
    result = np.empty(len(max_words), dtype=dtype)
    for i in range(len(max_words)):
        result[i] = max_words[i]
    return result





@njit
def lattice_dictionary_by_size(m:int):
    if m<=0 or m>MaximalWordLimit:
        raise ValueError(f"The number of maximal words is larger than {MaximalWordLimit}")
    word_type=WORD_TYPE #set_word_type(m)
    total=1<<m # 2**m
    lattice_words = np.arange(total, dtype=word_type)
    sizes=np.array([count_bits(x) for x in lattice_words], dtype=SizeType)
    d = Dict.empty(key_type=KEY_TYPE, value_type=VALUE_TYPE)
    for k in range(0,m+1):
        d[k]=lattice_words[sizes==k]
    return d

@njit
def nerve_of_max_words(maximal_words): 
    """ nerve_faces, intersection_list =nerve_of_max_words(maximal_words)
    Args:
        maximal_words (numpy.ndarray): a numpy array of maximal words

    Returns:
        nerve_dict: a dictionary of the nerve of maximal words
    """
    m = maximal_words.shape[0]
    if m > MaximalWordLimit:
        raise ValueError(f"The number of maximal words is larger than {MaximalWordLimit}")
    nerve_word_type = WORD_TYPE # set_word_type(m)
    # Compute the Boolean lattice on the maximal words.
    lattice_dictionary = lattice_dictionary_by_size(m)
    # Initialize faces with the vertices of the nerve (they are always in the nerve).
    faces = NumbaList([nerve_word_type(1 << j) for j in range(m)]) 
    intersection_list = NumbaList(maximal_words)
    minimal_non_faces = NumbaList.empty_list(WORD_TYPE_NUMBA )
    for k in range(2, m + 1):
        number_of_non_faces_in_k = 0
        for x in lattice_dictionary[k]:
            if x_is_a_superset_of_any_in_List(x, minimal_non_faces):
                number_of_non_faces_in_k += 1
                continue
            else:
                # Check if x is a face.
                intersection = intersection_of_codewords_from_bits(x, maximal_words)
                if intersection > 0:  # if the intersection is not empty, add x to faces.
                    faces.append(x)
                    intersection_list.append(intersection)
                else:  # if the intersection is empty, add x to minimal non-faces.
                    minimal_non_faces.append(x)
                    number_of_non_faces_in_k += 1
        if number_of_non_faces_in_k >= lattice_dictionary[k].shape[0]:
            break  # Stop searching for faces, since there no faces left to find.
    return faces,intersection_list 


@njit
def simplicial_violators_from_words(words: np.ndarray, maximal_words: np.ndarray) -> NumbaList:
    """ simplicial_violators_from_words(words, maximal_words)
    Args:
        words (numpy.ndarray): a numpy array of words
        maximal_words (numpy.ndarray): a numpy array of maximal words
    Returns:
        simplicial_violators: a list of simplicial violators
    """
    nerve_faces, intersection_list =nerve_of_max_words(maximal_words)
    # Convert the typed list 'intersection_list' to a NumPy array.
    n = len(intersection_list)
    tmp = np.empty(n, dtype=WORD_TYPE)  # WORD_TYPE is your NumPy type, e.g. np.uint32
    for i in range(n):
        tmp[i] = intersection_list[i]
    unique_intersections = np.unique(tmp)
    simplicial_violators = NumbaList.empty_list(WORD_TYPE_numba)
    for x  in  unique_intersections: 
        if not (x in words):
            simplicial_violators.append(x)
    return simplicial_violators









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
    def __init__(self, array_of_vectors: List[List[int]]):
        if not array_of_vectors:
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
            B=convert_to_boolean_matrix(array_of_vectors)
            n_bits=B.shape[1]
            self.dtype= WORD_TYPE# set_word_type(n_bits)
            words=np.unique(boolean_matrix_to_array_of_words(B,self.dtype))
            sizes=sizes_of_words(words)
            self.unique_sizes, indices = np.unique(sizes, return_inverse=True)
            indices_by_size=Dict.empty(key_type=types.int64, value_type=array_type)
            for (i,u) in enumerate(self.unique_sizes):
                indices_by_size[int(u)]=   np.where(indices == i)[0]
            self.indices_by_size=indices_by_size
            self.words=words
            self.sizes=sizes
            self.n_bits=B.shape[1]
            self.n_words=words.shape[0]
            self.min_size, self.max_size = self.unique_sizes.min(), self.unique_sizes.max()
            self.maximal_words= find_maximal_words(words, self.unique_sizes, indices_by_size,self.dtype)
    def has_empty_set(self):
        return bool(self.min_size==0)
    def has_full_set(self):
        return bool(self.max_size==self.n_bits)
    def simplicial_violators(self):
        return simplicial_violators_from_words(self.words, self.maximal_words)
    def Obstructions(self):
        return Obstructions(self)
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



###############  TDA utilities ############################

def compute_homology_from_facets(facets, max_dimension=np.inf):
    """
    Compute the persistence diagram and count the number of infinite bars by dimension.
    persistence, infinite_bar_counts = compute_homology_from_facets(facets, max_dimension=2)
    """

    st = gudhi.SimplexTree() # initializes a simplex tree

    # Insert each facet; Gudhi automatically adds lower-dimensional faces.
    for facet in facets:
        st.insert(facet)

    # Ensure the filtration is non-decreasing. 
    st.make_filtration_non_decreasing()
    if not np.isinf(max_dimension):
        st.prune_above_dimension(max_dimension) # remove simplices above given dimension, if we don't want to compute all possible homology groups

    persistence = st.persistence(persistence_dim_max=True) # persistence_dim_max=True tells to compute all homology groups up to maximal dimension

    # Extract infinite bars (features with death == infinity).
    infinite_bars = [(dim, (birth, death))
                     for dim, (birth, death) in persistence if death == float("inf")]
    
     # Count infinite bars by dimension.
    infinite_bar_counts = {}
    for dim, _ in infinite_bars:
        infinite_bar_counts[dim] = infinite_bar_counts.get(dim, 0) + 1
    
    return persistence, infinite_bar_counts


def homology_is_trivial(facets, max_dimension=np.inf): 
    """
    Check if the homology of the simplicial complex defined by the facets is that of a contractible space.
    homology_is_trivial(facets)

    Example:        
    facets = [[0, 1], [1, 2],[2, 3], [0, 3], [0,5], [5,6],[6,0], [10,11]]
    homology_is_trivial(facets) # False (the space is not contractible)
    """
    if len(facets) == 0:
        raise ValueError("Facets must be non-empty.")
    
    st = gudhi.SimplexTree() # initializes a simplex tree

    # Insert each facet; Gudhi automatically adds lower-dimensional faces.
    for facet in facets:
        st.insert(facet)

    # Ensure the filtration is non-decreasing. 
    st.make_filtration_non_decreasing()
    if not np.isinf(max_dimension):
        st.prune_above_dimension(max_dimension) # remove simplices above given dimension, if we don't want to compute all possible homology groups

    st.persistence(persistence_dim_max=True) # persistence_dim_max=True tells to compute all homology groups up to maximal dimension
    Betti_numbers = st.betti_numbers()
    return (Betti_numbers[0] == 1) and (sum(Betti_numbers[1:])==0)

