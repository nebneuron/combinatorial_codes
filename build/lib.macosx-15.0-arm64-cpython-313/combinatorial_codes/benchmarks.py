from combinatorial_codes import *
import numpy as np
C=example_code("example by Milo")
sv=C.simplicial_violators()
expected_violators = np.array([
            4, 5, 8, 12, 16, 20, 32, 64, 65,
            144, 256, 528, 1040, 1536, 2048, 2052, 2056, 2320,
            18432
        ], dtype=np.uint64)
correctness= np.array_equal(np.sort(sv), np.sort(expected_violators))

print("Simplicial violators correctness:", correctness)


the_path='/Users/vui1/Documents/GitHub/combinatorial_codes/src/combinatorial_codes'
import sys
sys.path.insert(0, the_path)
from utils import * 
import translated_functions as tf
lattice_slice2 = tf.lattice_slice2
# %%
from combinatorial_codes.examples import bernoulli_random_code
import cProfile
import pstats
import io
import line_profiler
import timeit

# %%
C = bernoulli_random_code(32, 35, 0.2)
print("number of maximal words=", len(C.maximal_words))
words = C.words
maximal_words = C.maximal_words
enforce_maximal_word_limit = False
m= len(maximal_words)
k=6

def run_intersection():
    intersection_list = intersections_via_cliques(maximal_words)
    return intersection_list

def run_lattice_slice2():
    minimal_non_faces = NumbaList([WORD_TYPE_NUMBA(999)])
    return tf.lattice_slice2(m, k, minimal_non_faces)
y=NumbaList(tf.lattice_slice2(m, k, minimal_non_faces))
x=lattice_slice(m, k, minimal_non_faces)





# Time the function: perform 5 runs, one execution per run.
times = timeit.repeat(run_lattice_slice, repeat=5, number=1)
print("Execution times over 5 runs:", times)
times = timeit.repeat(run_lattice_slice2, repeat=5, number=1)
print("Execution times over 5 runs:", times)

# Run the function with profiling.
profile = line_profiler.LineProfiler(lattice_slice2)
minimal_non_faces=NumbaList([WORD_TYPE_NUMBA(999)])
violators2 = profile.runcall(lattice_slice2, 28, 6,  minimal_non_faces)

# Print the detailed line-by-line timing report.
profile.print_stats()



# Optional: Also run cProfile for a higher-level overview.
pr = cProfile.Profile()
pr.enable()
violators = simplicial_violators_from_words(words, maximal_words, enforce_maximal_word_limit)
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime")
ps.print_stats("simplicial_violators_from_words")
print("cProfile stats:")
print(s.getvalue())

violators=C.simplicial_violators(enforce_maximal_word_limit=False)
print("number of simplicial violators=",len(violators))