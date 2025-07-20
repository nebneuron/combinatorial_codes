# Testing Guide for combinatorial_codes

This document explains how to run and understand the test suite for the combinatorial_codes package.

## Quick Start

Run all tests:
```bash
python run_tests.py
# or 
pytest tests/ -v
```

Run only the Milo example tests (the specific ones requested):
```bash
python run_tests.py --milo
```

## Test Structure

The test suite is organized into several test classes:

### `TestExampleCodes`
Tests for the example codes and their expected properties.

- **`test_milo_example_simplicial_violators`**: ✅ Verifies that `C.simplicial_violators()` returns the expected array of 19 values
- **`test_milo_example_obstructions`**: ✅ Verifies that `C.Obstructions()` returns `(False, 17)`
- **`test_milo_example_consistency`**: Tests basic consistency properties

### `TestOtherExampleCodes`
Basic functionality tests for other example codes (eyes, open not closed, closed not open).

### `TestCombinatorialCodeClass`
Core functionality tests for the CombinatorialCode class.

### `TestRandomCodes`
Tests for randomly generated codes using `bernoulli_random_code`.

### `TestPackageIntegration`
Tests for package-level features like C extension status checking.

## Key Test Requirements (Your Original Request)

The main tests you requested are:

1. **Simplicial Violators Test**:
   ```python
   C = example_code("example by Milo")
   C.simplicial_violators() == array([4, 5, 8, 12, 16, 20, 32, 64, 65,
                                     144, 256, 528, 1040, 1536, 2048, 2052, 2056, 2320,
                                     18432], dtype=uint64)
   ```

2. **Obstructions Test**:
   ```python
   C.Obstructions() == (False, 17)
   ```

Both of these tests **PASS** ✅

## Running Tests

### Option 1: Using the test runner script
```bash
# All tests
python run_tests.py

# Only Milo example tests
python run_tests.py --milo

# Fast tests only
python run_tests.py --fast

# With coverage report
python run_tests.py --coverage
```

### Option 2: Using pytest directly
```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_combinatorial_codes.py::TestExampleCodes::test_milo_example_simplicial_violators -v

# With coverage
pytest tests/ --cov=combinatorial_codes --cov-report=html
```

## Test Output Example

```
===================================== test session starts =====================================
platform darwin -- Python 3.13.3, pytest-8.4.1, pluggy-1.6.0
cachedir: .pytest_cache
rootdir: /Users/vui1/Documents/GitHub/combinatorial_codes
configfile: pytest.ini
plugins: cov-6.2.1
collected 12 items

tests/test_combinatorial_codes.py::TestExampleCodes::test_milo_example_simplicial_violators PASSED [  8%]
tests/test_combinatorial_codes.py::TestExampleCodes::test_milo_example_obstructions PASSED [ 16%]
tests/test_combinatorial_codes.py::TestExampleCodes::test_milo_example_consistency PASSED [ 25%]
...
================================ 12 passed, 1 warning in 2.09s ================================
```

## Test Dependencies

The tests require:
- `pytest >= 6.0`
- `pytest-cov` (for coverage reports)
- The `combinatorial_codes` package installed in development mode

These are automatically installed with:
```bash
pip install -e ".[test]"
```

## Adding New Tests

To add new tests:

1. Add test methods to the appropriate test class in `tests/test_combinatorial_codes.py`
2. Follow the naming convention: `test_*`
3. Use descriptive docstrings
4. Include both positive and negative test cases
5. Use appropriate assertions with clear error messages

Example:
```python
def test_my_new_feature(self):
    """Test description of what this test verifies."""
    # Arrange
    C = example_code("eyes")
    
    # Act
    result = C.my_new_method()
    
    # Assert
    assert result == expected_value, f"Expected {expected_value}, got {result}"
```

## Performance Testing

The package includes both C extensions and Numba fallbacks. Tests verify:
- Both performance modes work correctly
- Results are consistent between C and Numba implementations
- Graceful fallback when C extensions are unavailable

## Continuous Integration

This test suite is designed to work in CI environments. All tests should:
- Complete within reasonable time limits
- Not require external dependencies beyond those in `setup.py`
- Provide clear error messages when they fail
- Be deterministic (no random failures)
