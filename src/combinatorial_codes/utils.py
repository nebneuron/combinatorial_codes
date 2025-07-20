""" This module contains utility functions that are all numba-compiled. Most of these functions are handling 
heavy-lift operations that are used in the main code.
"""
import time
import numpy as np
import networkx as nx
from numba import jit, njit, types
from numba.typed import List as NumbaList
from numba.typed import Dict
from numba import typed
from numba.np.numpy_support import from_dtype
try:
    from . import translated_functions as tf # here we import the C-compiled functions from translated_functions
except ImportError:
    tf = None  # C-compiled functions not available, use pure Python/Numba implementations
# First, we define some constants and types used throughout the package.
MaximalWordLimit=64; # this is currently the maximal number of maximal words in a code that we can handle.
# The maximal number of bits in a word is 64,  
# This is  because the size of the lattice is 2**m, and we need to store the lattice in memory.


# Precompute some numba types that we will use later
array_type = types.Array(types.int64, 1, 'C') # a Numba array type: 1D float64 array in C-contiguous layout.
WORD_TYPE=np.uint64
MAX_NUMBER_OF_BITS = np.iinfo(WORD_TYPE).bits
WORD_TYPE_NUMBA = from_dtype(np.dtype(WORD_TYPE))
VALUE_TYPE = types.Array(from_dtype(np.dtype(WORD_TYPE)), 1, 'C')
SizeType=np.uint8 # no more than size =64 is supported for computing nerves of maximal words
empty_numba_list= NumbaList.empty_list(WORD_TYPE_NUMBA) # an empty numba list of WORD_TYPE_NUMBA


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


@njit
def bit_order(a: int, b: int) -> bool:
    # Returns True if every bit in a is <= every bit in b (i.e., if a's 1s are a subset of b's 1s)
    return (a & ~b) == 0

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
def intersection_of_codewords_from_bits_older(x, a):
    """
    Given:
      - x: an  unsigned integer
      - a: a NumPy vector of numbers (with length >= the number of set bits in x)
    Returns a single numpy unsigned integer that represents the intersection of the codewords
    in a that correspond to the bits set in x.
    """
    if x == 0:
        raise ValueError("The input x must be a positive integer")

    result = WORD_TYPE(-1)  # Initialize with all bits set (assuming bitwise AND)
    encountered_set_bit = False
    bit_index = 0
    temp_x = x

    while temp_x > 0:
        if temp_x & 1:  # Check if the current least significant bit is set
            if bit_index >= len(a):
                raise IndexError("The length of 'a' should be at least the number of set bits encountered in 'x'")

            if not encountered_set_bit:
                result = a[bit_index]
                encountered_set_bit = True
            else:
                result &= a[bit_index]

        temp_x >>= 1  # Right shift to check the next bit
        bit_index += 1

    if not encountered_set_bit:
        # Handle the case where x has no set bits after the loop (shouldn't happen if initial check passes)
        raise ValueError("Input 'x' did not have any set bits.")
    return WORD_TYPE(result) # Explicitly cast the result to a NumPy unsigned integer


@jit(nopython=True)
def intersection_of_codewords_from_bits(x, a):
    """
    Given:
      - x: an  unsigned integer
      - a: a NumPy vector of numbers (with length >= the number of set bits in x)
    Returns a single numpy unsigned integer that represents the intersection of the codewords
    in a that correspond to the bits set in x. Returns 0 immediately if the intersection becomes 0.
    """
    if x == 0:
        raise ValueError("The input x must be a positive integer")

    result = np.iinfo(WORD_TYPE).max # Initialize with all bits set (assuming bitwise AND)
    encountered_set_bit = False
    bit_index = 0
    temp_x = x
    while temp_x > 0:
        if result == 0:  # Early exit optimization
            return np.uint64(0)

        if temp_x & 1:  # Check if the current least significant bit is set
            if bit_index >= len(a):
                raise IndexError("The length of 'a' should be at least the number of set bits encountered in 'x'")

            if not encountered_set_bit:
                result = a[bit_index]
                encountered_set_bit = True
            else:
                result &= a[bit_index]

        temp_x >>= 1  # Right shift to check the next bit
        bit_index += 1

    if result == WORD_TYPE(-1) and x > 0: # Handle case where x has set bits but result remained at initial value (first bit set scenario)
        return np.uint64(a[0] if len(a) > 0 and (x & 1) else 0) # Return first element if only first bit was set
    return np.uint64(result)



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






@jit(nopython=True) # here we optimize for speed, as this function is a heavy lifter
def x_is_a_superset_of_any_in_List(x: type, L : NumbaList) -> bool:
    """This evaluates if any word in the list L is a subset of x
        here we interpret both x and elements of L as sets encoded as bitarrays.
    Args:
        x : dtype
        L (NumbaList): a list of words of type dtype
    Returns: True if x is a superset of any of the words in L
    usage:
           logical=x_is_a_subset_of_any_in_List(x,L)
    """
    if len(L)==0:
        return False
    for b in L:
            if ( b & ~( type(b)(x))) == 0:
                return True
    return False









@njit
def binom(n, k):
    """Compute the binomial coefficient C(n,k) using integer arithmetic."""
    if k > n:
        return 0
    if k == 0 or k == n:
        return 1
    result = 1
    for i in range(1, k+1):
        result = result * (n - i + 1) // i
    return result

@njit
def generate_increasing_tuples_nb(m, k):
    """
    Generate all k-tuples (a1, a2, ..., ak) such that:
      - Each a_i is in range(0, m)
      - a1 < a2 < ... < ak (strictly increasing order)
      
    Returns a 2D NumPy array where each row is one tuple.
    """
    ncomb = binom(m, k)
    result = np.empty((ncomb, k), dtype=SizeType)
    
    # 'current' holds the current combination in increasing order.
    current = np.arange(k, dtype=SizeType)

    row_index = 0
    done = False
    while not done:
        # Store the current combination (already in strictly increasing order).
        for j in range(k):
            result[row_index, j] = current[j]
        row_index += 1

        # Find the rightmost index that can be incremented.
        i = k - 1
        while i >= 0 and current[i] == m - k + i:
            i -= 1
        if i < 0:
            done = True  # All combinations have been generated.
        else:
            current[i] += 1
            for j in range(i + 1, k):
                current[j] = current[j - 1] + 1
    return result


@njit
def lattice_slice(m:int, k:int, minimal_non_faces: NumbaList) -> np.ndarray:
    """
    Inputs: 
    m is the number of vertices in the lattice, 
    k is the number of bits in each word.
    The function returns a numpy array of integers of type WORD_TYPE.
    """
    # first we check the sanity of the inputs 
    if m>MaximalWordLimit:
        raise ValueError(f"The number m={m} of maximal words is larger than MaximalWordLimit={MaximalWordLimit}")
    elif k<=1:
        raise ValueError(f"The size k={k} is not larger than 1")
    elif  k>m:
        raise ValueError(f"The size k={k} is larger than m={m}")
    # precompute the shift table
    shift_table=np.array( [1<<i for i in range(m)]  ,dtype=WORD_TYPE)
    tuples=generate_increasing_tuples_nb(m,k)
    words=NumbaList.empty_list(WORD_TYPE_NUMBA )
    for t in tuples:
        x = shift_table[t].sum()
        if not x_is_a_superset_of_any_in_List(x, minimal_non_faces):
            words.append(WORD_TYPE(x))
    # Convert NumbaList to numpy array
    n = len(words)
    result = np.empty(n, dtype=WORD_TYPE)
    for i in range(n):
        result[i] = words[i]
    return result



@njit(cache=True)
def lattice_slice_from_sigma(
    sigma: np.ndarray,  # 1‑D array of *global* vertex indices (uint64)
    k: int,             # 1 ≤ k ≤ len(sigma)
    minimal_non_faces,  # numba.typed.List[WORD_TYPE] – forbidden masks (global)
):
    """Enumerate every k‑subset of ``sigma`` that avoids the forbidden masks.
    Usage: 
    faces=lattice_slice_from_sigma(sigma, k, minimal_non_faces)
    Returns
    -------
    np.ndarray[WORD_TYPE] -- 1‑D array of codewords (bit masks).
    """
    m = sigma.size
    if k < 1 or k > m:
        raise ValueError("k must satisfy 1 ≤ k ≤ len(sigma)")

    # Pre‑compute 1 << v for every vertex label in sigma (global positions)
    shifts = np.empty(m, dtype=WORD_TYPE)
    for idx in range(m):
        shifts[idx] = WORD_TYPE(1) << sigma[idx]

    tuples = generate_increasing_tuples_nb(m, k)
    words = NumbaList.empty_list(WORD_TYPE_NUMBA)

    for t in tuples:
        mask = WORD_TYPE(0)
        for idx in t:
            mask |= shifts[idx]

        keep = True
        for forbidden in minimal_non_faces:
            if (mask & forbidden) == forbidden:
                keep = False
                break
        if keep:
            words.append(WORD_TYPE(mask))

    # Copy to a contiguous NumPy array for the caller
    return np.asarray(words, dtype=WORD_TYPE)


@njit(cache=True)
def lattice_slice_from_sigma2(
    sigma: np.ndarray,             # 1-D array of vertex labels
    k: int,                        # 1 ≤ k ≤ len(sigma)
    minimal_non_faces1,            # numba.typed.List[np.uint64]
    minimal_non_faces2: np.ndarray # 1-D ndarray[np.uint64]
) -> np.ndarray:
    """Return every k-face of ``sigma`` that avoids all masks in *both* lists."""
    m = sigma.size
    if k < 1 or k > m:
        raise ValueError("k must satisfy 1 ≤ k ≤ len(sigma)")

    # Precompute 1 << label for each vertex label
    shifts = np.empty(m, dtype=np.uint64)
    for i in range(m):
        shifts[i] = np.uint64(1) << np.uint64(sigma[i])

    tuples = generate_increasing_tuples_nb(m, k)

    words = typed.List.empty_list(np.uint64)  # type: ignore[var-annotated]
    n2 = minimal_non_faces2.size

    for t in tuples:
        mask = np.uint64(0)
        for idx in t:
            mask |= shifts[idx]

        keep = True
        for forb in minimal_non_faces1:       # first non-face list
            if (mask & forb) == forb:
                keep = False
                break
        if not keep:
            continue

        for j in range(n2):                  # second non-face list
            forb = minimal_non_faces2[j]
            if (mask & forb) == forb:
                keep = False
                break

        if keep:
            words.append(mask)

    return np.asarray(words, dtype=np.uint64)




@njit
def intersection_graph(maximal_words: np.ndarray):
    """ 
    Args:
    maximal_words (numpy.ndarray): a numpy array of maximal words
    Returns: 
    a list of edges of the intersection graph of the maximal words.
    The intersection graph is a graph where each maximal word is a vertex, and there is an edge 
    between two vertices if their intersection is non-empty.
    Here each vertex is the index of the appropriate maximal word in the maximal_words array.
    The edges are represented as tuples of two indices (i, j) where i < j.
    The function returns a NumbaList of tuples, where each tuple is of type int64[2].
    Usage: edges = intersection_graph(maximal_words)    
    """
    m = maximal_words.shape[0]
    edges = NumbaList()
    for i in range(m):
        for j in range(i + 1, m):
            if (maximal_words[i] & maximal_words[j]) > 0:
                edges.append((i, j))
    return edges


@njit 
def intersections_inside_a_clique(maximal_words: np.ndarray, clique: np.ndarray, minimal_non_faces: NumbaList) -> np.ndarray:
    """ Usage:
        unique_intersections =intersections_inside_a_clique(maximal_words, clique, minimal_non_faces)
    Args:
        maximal_words (numpy.ndarray): a numpy array of maximal words
        clique (numpy.ndarray): a numpy array of indices of the maximal words 
        minimal_non_faces (NumbaList): a list of forbidden faces (non-faces) of the nerve
        Note that the indices in clique are global indices of the maximal_words array.
        ** Importantly, this function modifies the minimal_non_faces list by adding new non-faces to it.
        
    """
    # here the minimal_non_faces is a list of words that are not faces of the nerve
    # note that  minimal_non_faces is represented in terms of the global indexing of maximal_words  
    m=len(clique)
    intersection_list = NumbaList.empty_list(WORD_TYPE_NUMBA)
    for k in range(2, m + 1):
        candidate_faces=lattice_slice_from_sigma(clique, k, minimal_non_faces) 
        # the array candidate_faces is a numpy array of integers of type WORD_TYPE
        # the candidate_faces are generated from the clique, and reflect the 
        # original indexing inside all the indicies of the maximal_words
        n_candidate_faces = len(candidate_faces)
        if n_candidate_faces == 0:
            break # No more faces of the nerve to check.
        rejected_face_number = 0
        for x in candidate_faces:
            # Check if x is a face.
            intersection = intersection_of_codewords_from_bits(x, maximal_words)
            if intersection !=0:  # if the intersection is not empty, add x to faces.
                intersection_list.append(WORD_TYPE(intersection))
            else:  # if the intersection is empty, add x to minimal non-faces.
                # Ensure x is properly cast to WORD_TYPE to avoid int64->uint64 warnings
                minimal_non_faces.append(WORD_TYPE(x))
                rejected_face_number += 1
        if rejected_face_number == n_candidate_faces:
            break  # Stop searching for faces, since there no faces left to find.
    # Remove duplicates from  the intersection list
    n = len(intersection_list)
    tmp = np.empty(n, dtype=WORD_TYPE)
    for i in range(n):
        tmp[i] = intersection_list[i]
    unique_intersections = np.unique(tmp)
    return unique_intersections
 
@njit 
def intersections_inside_a_clique_optimized(maximal_words: np.ndarray, clique: np.ndarray, minimal_non_faces: np.ndarray) -> np.ndarray:
    """ Usage:
        unique_intersections, new_non_faces =intersections_inside_a_clique_optimized(maximal_words, clique, minimal_non_faces)
    Args:
        maximal_words (numpy.ndarray): a numpy array of maximal words
        clique (numpy.ndarray): a numpy array of indices of the maximal words 
        minimal_non_faces (NumbaList): a list of forbidden faces (non-faces) of the nerve
        Note that the indices in clique are global indices of the maximal_words array.
        This version does NOT  modify the minimal_non_faces list. Instead, 
        it returns a new list of non-faces that are found during the search.
        
    """
    # here the minimal_non_faces is a list of words that are not faces of the nerve
    # note that  minimal_non_faces is represented in terms of the global indexing of maximal_words  
    m=len(clique)
    intersection_list = NumbaList.empty_list(WORD_TYPE_NUMBA)
    new_non_faces = NumbaList.empty_list(WORD_TYPE_NUMBA)
    for k in range(2, m + 1):
        candidate_faces=lattice_slice_from_sigma2(clique, k, new_non_faces, minimal_non_faces) 
        # the array candidate_faces is a numpy array of integers of type WORD_TYPE
        # the candidate_faces are generated from the clique, and reflect the 
        # original indexing inside all the indicies of the maximal_words
        n_candidate_faces = len(candidate_faces)
        if n_candidate_faces == 0:
            break # No more faces of the nerve to check.
        rejected_face_number = 0
        for x in candidate_faces:
            # Check if x is a face.
            intersection = intersection_of_codewords_from_bits(x, maximal_words)
            if intersection !=0:  # if the intersection is not empty, add x to faces.
                intersection_list.append(WORD_TYPE(intersection))
            else:  # if the intersection is empty, add x to minimal non-faces.
                # Ensure x is properly cast to WORD_TYPE to avoid int64->uint64 warnings
                new_non_faces.append(WORD_TYPE(x))
                rejected_face_number += 1
        if rejected_face_number == n_candidate_faces:
            break  # Stop searching for faces, since there no faces left to find.
    # Remove duplicates from  the intersection list
    n = len(intersection_list)
    tmp = np.empty(n, dtype=WORD_TYPE)
    for i in range(n):
        tmp[i] = intersection_list[i]
    unique_intersections = np.unique(tmp)
    return unique_intersections, new_non_faces
 



def intersections_work(maximal_words: np.ndarray, maximal_cliques: list[np.ndarray], max_cliques_as_words: np.ndarray) -> np.ndarray:
    # Initialize the intersection minimal_non_faces lists
    intersection_list = NumbaList.empty_list(WORD_TYPE_NUMBA)
    minimal_non_faces = NumbaList.empty_list(WORD_TYPE_NUMBA)
    m = maximal_words.shape[0]
    m_mask = (WORD_TYPE(1) << m) - 1
    for i,clique in enumerate(maximal_cliques):
        clique_as_word= max_cliques_as_words[i]
        clique_complement_as_word = ~clique_as_word & m_mask
        if len(minimal_non_faces) > 0:
            # here we compose the list of _relevant_for_this_clique_ minimal non-faces
            # Optimized version: single pass through minimal_non_faces, build relevant list directly
            relevant_minimal_non_faces_list = NumbaList.empty_list(WORD_TYPE_NUMBA)
            for x in minimal_non_faces:
                if (x & clique_complement_as_word) == 0:
                    relevant_minimal_non_faces_list.append(x)
            
            # Convert to numpy array only once
            n_relevant = len(relevant_minimal_non_faces_list)
            relevant_minimal_non_faces = np.empty(n_relevant, dtype=WORD_TYPE)
            for i in range(n_relevant):
                relevant_minimal_non_faces[i] = relevant_minimal_non_faces_list[i]
        else:
            relevant_minimal_non_faces = np.empty(0, dtype=WORD_TYPE)  #
        intersections_this_clique, new_non_faces =intersections_inside_a_clique_optimized(maximal_words, clique,relevant_minimal_non_faces)
        intersection_list.extend(intersections_this_clique)
        minimal_non_faces.extend(new_non_faces)
    
    # now we remove duplicates from the intersection list
    n = len(intersection_list)
    result = np.empty(n, dtype=WORD_TYPE)
    for i in range(n):
        result[i] = intersection_list[i]
    # result is a sorted array of  unique intersections, that are not maximal words
    # note that the result does NOT contain the empty set (encoded by a zero)
    result = np.unique(result) 
    return result




def intersections_via_cliques(maximal_words: np.ndarray) -> np.ndarray: 
    """ 
    Args:
    maximal_words (numpy.ndarray): a numpy array of maximal words
    Returns: 
    intersection_list = intersections_via_cliques(maximal_words)
    intersection_list: an  array of unique intersections of the maximal words, 
    this does not include zero (the empty set), neither  does it include the maximal words, 
    as those are not simplicial violators. 
    """
    if maximal_words.shape[0] > MaximalWordLimit:
        raise ValueError(f"The number of maximal words {maximal_words.shape[0]} is larger than MaximalWordLimit={MaximalWordLimit}. Currently we cannot handle more than {MaximalWordLimit} maximal words.   ")
    G = nx.Graph()
    G.add_edges_from(intersection_graph(maximal_words))
    maximal_cliques = list(nx.find_cliques(G))
    # sort the maximal cliques by size
    maximal_cliques.sort(key=len, reverse=True)
    # Convert to a NumPy array
    maximal_cliques = [np.array(sorted(q), dtype=np.uint64) for q in maximal_cliques]
    max_cliques_as_words,_=convert_to_array_of_words(maximal_cliques)
    # check if tf is working 
    if tf is not None:
        print("Using C-compiled functions for intersections.")
        return tf.intersections_work(maximal_words, maximal_cliques, max_cliques_as_words)
    else:
        print("C-compiled functions are not available, using pure Python/Numba implementation.")
        return intersections_work(maximal_words, maximal_cliques, max_cliques_as_words)

def simplicial_violators_from_words(words: np.ndarray, maximal_words: np.ndarray, enforce_maximal_word_limit: bool=True) -> np.ndarray:
    """ simplicial_violators_from_words(words, maximal_words)
    Args:
    words (numpy.ndarray): a numpy array of words
    maximal_words (numpy.ndarray): a numpy array of maximal words
    Returns:
        simplicial_violators: a numpy array of simplicial violators
    """
    # first, make sure that words is sorted 
    sorted_words = words if (np.all(words[:-1] < words[1:])) else np.sort(words)
    n_words=len(sorted_words)
    unique_intersections = intersections_via_cliques(maximal_words)
    n_uv=unique_intersections.shape[0]
    is_violator=np.empty(n_uv, dtype=np.bool_)
    # we can assume that unique_intersections are sorted, since they were obtained from np.unique
    # we check if each unique intersection is contained in the list of words
    indx_words=0
    # the following is inelegant, but is the fastest way to do it 
    for i,x  in  enumerate(unique_intersections): 
        while indx_words<n_words and sorted_words[indx_words]<x:
                indx_words+=1
        if indx_words==n_words: # this means that all the words are smaller than x, x is a violator 
                                # and all intersections that are left are also violators
            is_violator[i:]=True
            break # no need to search further
        is_violator[i] = (sorted_words[indx_words] != x)
    return unique_intersections[is_violator]




