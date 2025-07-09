from combinatorial_codes import *
# %%
from combinatorial_codes.examples import bernoulli_random_code
import cProfile
import pstats
import io
import line_profiler

# %%
C = bernoulli_random_code(32, 28, 0.3)
print("number of maximal words=", len(C.maximal_words))
words = C.words
maximal_words = C.maximal_words
enforce_maximal_word_limit = False

# Example: Use line_profiler to dive inside simplicial_violators_from_words.
# Register the target function (replace with the correct function if named differently)
profile = line_profiler.LineProfiler(lattice_slice)

# Run the function with profiling.
violators = profile.runcall(lattice_slice, 28, 6,  minimal_non_faces)

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