# Combinatorial Codes

A Python package for identifying obstructions to convexity in combinatorial codes, as described in:

> J. Cruz, C. Giusti, V. Itskov, B. Kronholm. **On open and closed convex codes.** *Discrete and Computational Geometry*, 61(2):247–270, 2019.

## Quick Install

```bash
pip install -e .
```

The package includes optional C extensions compiled automatically during installation for a 5–10× speedup on large codes. If compilation fails, the package falls back to a Numba-based implementation and remains fully functional.

## Usage

### Creating a code

```python
from combinatorial_codes import CombinatorialCode

code = CombinatorialCode([[], [1], [2], [3], [1, 2], [2, 3], [1, 3]])
print(code)
```

### Identifying obstructions to convexity

```python
is_intersection_complete, num_obstructions = code.Obstructions()
print(is_intersection_complete, num_obstructions)
```

### Finding simplicial violators

```python
violators = code.simplicial_violators()
print(violators)
```

### Built-in example codes

```python
from combinatorial_codes import example_code

code1 = example_code('eyes')            # non-convex code on three neurons
code2 = example_code('closed not open') # closed convex but not open convex
code3 = example_code('open not closed') # open convex but not closed convex
```

### Generating a random Bernoulli code

```python
from combinatorial_codes import bernoulli_random_code

code = bernoulli_random_code(n_bits=9, Nwords=100, p=0.25)
print(code)
```

## Testing

```bash
pytest tests/ -v
```

See [TESTING.md](TESTING.md) for details.

## License

MIT License.

## Citation

```bibtex
@article{convexcodes2019,
  title   = {On Open and Closed Convex Codes},
  author  = {Joshua Cruz and Chad Giusti and Vladimir Itskov and Bill Kronholm},
  journal = {Discrete and Computational Geometry},
  year    = {2019},
  volume  = {61},
  number  = {2},
  pages   = {247--270},
  publisher = {Springer New York},
}
```
