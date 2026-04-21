# Combinatorial Codes

A Python package for studying combinatorial codes and identifying obstructions to convexity.

Primary reference:

> J. Cruz, C. Giusti, V. Itskov, B. Kronholm. **On Open and Closed Convex Codes.**
> *Discrete and Computational Geometry* 61(2):247-270, 2019.

## Install

```bash
pip install -e .
```

Optional C extensions are built automatically. If compilation fails, the package falls back to a Numba-based implementation and remains usable.

For installation verification, build requirements, and C extension troubleshooting, see [INSTALLATION_AND_COMPILATION.md](INSTALLATION_AND_COMPILATION.md).

For a quick verification after install:

```bash
python getting_started.py
```

## Usage

### Create a code

```python
from combinatorial_codes import CombinatorialCode

code = CombinatorialCode([[], [1], [2], [3], [1, 2], [2, 3], [1, 3]])
print(code)
```

### Check basic properties

```python
print(code.has_empty_set())
print(code.has_full_set())
```

### Compute obstruction information

```python
is_intersection_complete, num_obstructions = code.Obstructions()
print(is_intersection_complete, num_obstructions)
```

### Find simplicial violators

```python
violators = code.simplicial_violators()
print(violators)
```

### Built-in example codes

```python
from combinatorial_codes import example_code

code1 = example_code("eyes")
code2 = example_code("closed not open")
code3 = example_code("open not closed")
```

### Random Bernoulli code

```python
from combinatorial_codes import bernoulli_random_code

code = bernoulli_random_code(n_bits=9, Nwords=100, p=0.25)
print(code)
```

## Testing

Run the test suite with:

```bash
pytest tests/ -v
```

See [TESTING.md](TESTING.md) for more detail.

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
