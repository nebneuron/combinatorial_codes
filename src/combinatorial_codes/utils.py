""" This module contains utility functions that are all numba-compiled. Most of these functions are handling 
heavy-lift operations that are used in the main code.
"""
import numpy as np
from numba import jit, njit, types
from numba.typed import List as NumbaList
from numba.typed import Dict
from numba.np.numpy_support import from_dtype

# First, we define some constants and types used throughout the package.
MaximalWordLimit=40; # this is currently the maximal number of maximal words in a code that we can handle.
# The maximal number of bits in a word is 64, but we can only handle 30 maximal words
# This is  because the size of the lattice is 2**m, and we need to store the lattice in memory.


# Precompute some numba types that we will use later
array_type = types.Array(types.int64, 1, 'C') # a Numba array type: 1D float64 array in C-contiguous layout.
WORD_TYPE=np.uint32
MAX_NUMBER_OF_BITS = np.iinfo(WORD_TYPE).bits
WORD_TYPE_NUMBA = from_dtype(np.dtype(WORD_TYPE))
VALUE_TYPE = types.Array(from_dtype(np.dtype(WORD_TYPE)), 1, 'C')
SizeType=np.uint8 # no more than size =64 is supported for computing nerves of maximal words





@njit
def bit_order(a: int, b: int) -> bool:
    # Returns True if every bit in a is <= every bit in b (i.e., if a's 1s are a subset of b's 1s)
    return (a & ~b) == 0






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
    The function returns a numpy array of integers of typr WORD_TYPE.
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
            words.append(x)
    return words



@njit
def intersections_list_2(maximal_words: np.ndarray) -> np.ndarray: 
    """ 
    Args:
    maximal_words (numpy.ndarray): a numpy array of maximal words
    Returns: 
    intersection_list = intersections_list_2(maximal_words)
    intersection_list: an  array of unique intersections of the maximal words, 
    this does not include zero (the empty set), neither  does it include the maximal words, 
    as those are not simplicial violators. 
    """
    m = maximal_words.shape[0]
    if m > MaximalWordLimit:
        raise ValueError(f"The number of maximal words = {m} is larger than {MaximalWordLimit}")
    nerve_word_type = WORD_TYPE # set_word_type(m)
    # Initialize the intersection the minimal_non_faces lists
    intersection_list = NumbaList.empty_list(WORD_TYPE_NUMBA)
    minimal_non_faces = NumbaList.empty_list(WORD_TYPE_NUMBA)
    for k in range(2, m + 1):
        candidate_faces = lattice_slice(m, k, minimal_non_faces)
        n_candidate_faces = len(candidate_faces)
        if n_candidate_faces == 0:
            break # No more faces of the nerve to check.
        rejected_face_number = 0
        for x in candidate_faces:
            # Check if x is a face.
            intersection = intersection_of_codewords_from_bits(x, maximal_words)
            if intersection > 0:  # if the intersection is not empty, add x to faces.
                intersection_list.append(intersection)
            else:  # if the intersection is empty, add x to minimal non-faces.
                minimal_non_faces.append(x)
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
def simplicial_violators_from_words(words: np.ndarray, maximal_words: np.ndarray) -> NumbaList:
    """ simplicial_violators_from_words(words, maximal_words)
    Args:
    words (numpy.ndarray): a numpy array of words
    maximal_words (numpy.ndarray): a numpy array of maximal words
    Returns:
        simplicial_violators: a list of simplicial violators
    """
    # first, make sure that words is sorted 
    sorted_words = words if (np.all(words[:-1] < words[1:])) else np.sort(words)
    n_words=len(sorted_words)
    unique_intersections  =intersections_list_2(maximal_words)
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









@njit
def lattice_dictionary_by_size(m:int):
    if m<=0 or m>MaximalWordLimit:
        raise ValueError(f"The number of maximal words is larger than {MaximalWordLimit}")
    total=1<<m # 2**m
    lattice_words = np.arange(total, dtype=WORD_TYPE)
    sizes=np.array([count_bits(x) for x in lattice_words], dtype=SizeType)
    d = Dict.empty(key_type=SizeType, value_type=VALUE_TYPE)
    for k in np.arange(0, m+1,dtype=SizeType):
        d[k] = lattice_words[sizes==k]
    return d

