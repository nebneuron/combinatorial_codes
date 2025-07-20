# __init__.py
__version__ = "0.2.0"

from .tda import homology_is_trivial
from .tda import compute_homology_from_facets
from .codes import CombinatorialCode
from .codes import WORD_TYPE
from .codes import boolean_matrix_to_array_of_words
from .codes import convert_to_boolean_matrix
from .codes import array_of_words_to_vectors_of_integers
from .utils import find_maximal_words
from .codes import Obstructions
from .utils import x_is_a_subset_of_any_in_List
from .utils import NumbaList
from .utils import binom
from .utils import count_bits
from .utils import custom_bit_length
from .utils import intersection_of_codewords_from_bits
from .utils import bit_order
from .codes import array_of_words_to_boolean_matrix
from .codes import array_of_words_to_vectors_of_integers
from .utils import link_facets
from .utils import x_is_a_superset_of_any_in_List
from .utils import intersections_via_cliques
from .utils import intersections_inside_a_clique
from .utils import simplicial_violators_from_words
from .examples import example_code
from .examples import example_dictionary
from .examples import bernoulli_random_code
from .status import check_c_extension_status, quick_status
# Import translated_functions (C extension) if available
try:
    from . import translated_functions
except ImportError:
    translated_functions = None

__all__ = [
    "CombinatorialCode", "Obstructions",
    "WORD_TYPE",
    "boolean_matrix_to_array_of_words",
    "convert_to_boolean_matrix",
    "find_maximal_words",
    "x_is_a_subset_of_any_in_List",
    "NumbaList", "homology_is_trivial", "compute_homology_from_facets", 
    "example_code", "example_dictionary","bernoulli_random_code",
    "simplicial_violators_from_words",
    "array_of_words_to_vectors_of_integers",
    "intersections_via_cliques",
    "intersections_inside_a_clique",
    "check_c_extension_status", "quick_status",
    "translated_functions"
]


