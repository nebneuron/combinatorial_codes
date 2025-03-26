# Combinatorial Codes

This package provides tools for manipulating combinatorial codes. It includes methods to identify obstructions to convexity in combinatorial codes.


## Installation

To install the package, do the following: 
* download the package
* change to the appropriate directory, as in
```
cd combinatorial_codes/
```
* use the following command from commandline, to install the package into an active python environment:

```
 pip install -e .
```

## Usage

### Creating a Combinatorial Code

You can create a combinatorial code by passing a list of lists of integers that represent the codewords:

```python
 from combinatorial_codes import CombinatorialCode, example_code

array_of_vectors = [[],[1],[2],[3],[1, 2], [2, 3], [1, 3]] # note that we include the empty set here. Currently we should always include empty set in a code. 
code = CombinatorialCode(array_of_vectors)
print(code)
```
### Computing the local obstruction information
```
is_maximal_intersection_complete, num_obstructions = code.Obstructions()
print(is_maximal_intersection_complete, num_obstructions)
```

### There are some example codes that can be acced via the  example_code(code_name) as in the following: 
```
code1=example_code('eyes')            # a non-convex code on three neurons that resembles eyes
code2=example_code('closed not open') # a "good" code that is not intersection complete
code3=example_code('open not closed') # a "good" code that is not intersection complete
```




### Checking for Empty and Full Sets

You can check if the code has an empty set or a full set:

```python
print(code.has_empty_set())  # True or False
print(code.has_full_set())   # True or False
```

### Finding Simplicial Violators

You can find the simplicial violators of the code:

```python
violators = code.simplicial_violators()
print(violators)
```

### Identifying Obstructions

You can identify obstructions to the code being a good code:

```python
is_maximal_intersection_complete, num_obstructions = code.Obstructions()
print(is_maximal_intersection_complete)
print(num_obstructions)
```

## Examples

You can use predefined examples from the `examples.py` file:

```python
from combinatorial_codes import example_code

example = example_code("eyes")
print(example)
```

## Dependencies

The package requires the following dependencies:

- `numba>=0.57.0`
- `numpy>=2.1.3`
- `gudhi>=3.11.0`

## License

This project is licensed under the MIT License.

## Citation

If you use this work or its underlying theory in your research, please cite the following paper:

For BibTeX users:
```
@article{convexcodes2019,
title = "On Open and Closed Convex Codes",
author = "Joshua Cruz and Chad Giusti and Vladimir Itskov and Bill Kronholm",
year = "2019",
volume = "61",
pages = "247--270",
journal = "Discrete and Computational Geometry",
publisher = "Springer New York",
number = "2",}
```
