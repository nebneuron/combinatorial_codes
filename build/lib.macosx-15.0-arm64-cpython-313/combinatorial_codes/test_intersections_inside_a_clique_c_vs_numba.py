# test_intersections_inside_a_clique_c_vs_numba.py
"""Compare the C translation against the original Numba implementation.

Requires that:
* The extension module *translated_functions* is built in-place.
* The original Numba version is importable as
  ``utils.intersections_inside_a_clique`` (adjust the import if different).

Run:
    pytest -q test_intersections_inside_a_clique_c_vs_numba.py
"""
from __future__ import annotations

import random
from typing import List

import numpy as np
import pytest

try:
    from .translated_functions import intersections_inside_a_clique as c_intersections_inside_a_clique
except ImportError:  # pragma: no cover – extension not compiled
    pytest.skip("translated_functions extension not built", allow_module_level=True)

try:
    from .utils import intersections_inside_a_clique as nb_intersections_inside_a_clique, WORD_TYPE, WORD_TYPE_NUMBA
except ImportError:  # pragma: no cover – reference missing
    pytest.skip("Numba reference implementation not importable", allow_module_level=True)

from numba import typed

# Ensure the Numba function's global namespace contains `typed` so it compiles
from . import utils as _utils_mod
if not hasattr(_utils_mod, "typed"):
    setattr(_utils_mod, "typed", typed)


RAND = random.Random(0)


def build_random_instance(max_len: int = 32, max_clique: int = 10):
    """Return (maximal_words, clique, minimal_non_faces) random but valid."""
    m = RAND.randint(5, max_len)
    maximal_words = np.array([RAND.getrandbits(64) | 1 for _ in range(m)], dtype=np.uint64)

    # Build clique as indices 0..m-1 shuffled and take subset
    idxs = list(range(m))
    RAND.shuffle(idxs)
    s = min(RAND.randint(3, max_clique), m)
    clique = np.array(idxs[:s], dtype=np.uint64)

    # Minimal non-faces – start empty
    mnf_list = typed.List.empty_list(WORD_TYPE_NUMBA)
    return maximal_words, clique, mnf_list


@pytest.mark.parametrize("_", range(50))
def test_random_instances(_):
    maximal_words, clique, mnf = build_random_instance()

    # Clone minimal_non_faces for each implementation because the function mutates it
    # Cannot use copy.deepcopy on NumbaList, so manually copy
    mnf_nb = typed.List.empty_list(WORD_TYPE_NUMBA)
    for item in mnf:
        mnf_nb.append(item)
    
    mnf_c = typed.List.empty_list(WORD_TYPE_NUMBA)
    for item in mnf:
        mnf_c.append(item)

    ref = nb_intersections_inside_a_clique(maximal_words, clique, mnf_nb)
    test = c_intersections_inside_a_clique(maximal_words, clique, mnf_c)

    # Sort for deterministic comparison
    assert np.array_equal(np.sort(ref), np.sort(test))

    # The minimal_non_faces lists should also match (order may differ)
    assert sorted(mnf_nb) == sorted(mnf_c)
