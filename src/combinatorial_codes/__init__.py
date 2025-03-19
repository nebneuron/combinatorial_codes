# __init__.py
from .codes import CombinatorialCode
from .codes import WORD_TYPE
from .codes import boolean_matrix_to_array_of_words
from .codes import sizes_of_words
from .codes import convert_to_boolean_matrix
from .codes import find_maximal_words
from .codes import x_is_a_subset_of_any_in_List
from .codes import NumbaList
from .codes import set_word_type

__all__ = [
    "CombinatorialCode",
    "WORD_TYPE",
    "set_word_type",
    "boolean_matrix_to_array_of_words",
    "sizes_of_words",
    "convert_to_boolean_matrix",
    "find_maximal_words",
    "x_is_a_subset_of_any_in_List",
    "NumbaList",
]


