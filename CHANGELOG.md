# Changelog

All notable changes to the combinatorial_codes package are documented in this file.

## [0.2.0] - 2025-01-26

### Added
- **`add_empty_word()` method**: New method in `CombinatorialCode` class to add the empty word (empty set) to a combinatorial code
  - Handles both empty codes and codes with existing words
  - Properly updates all code attributes (`unique_sizes`, `indices_by_size`, `min_size`, etc.)
  - Returns boolean indicating if empty word was added or already existed
  
- **Post-installation verification system**: Automatic verification runs during `pip install`
  - New `install_verification.py` module with comprehensive regression tests
  - `verify_installation()` function for manual verification
  - Tests package imports, C extension status, and computational correctness
  - Validates against reference data for "random example 32"
  - Performance benchmarking of key operations
  
- **Regression testing framework**: 
  - New `correct_examples.py` with reference results for "random example 32"
  - `RANDOM_EXAMPLE_32_INTERSECTIONS`: 494 intersection values
  - `RANDOM_EXAMPLE_32_OBSTRUCTIONS`: (False, 426) obstruction results
  - Unit tests in `test_combinatorial_codes.py` validate against reference data

- **Enhanced package distribution**:
  - Test files now included in pip installations via `MANIFEST.in`
  - `CustomInstall` class in `setup.py` runs post-installation verification
  - Package data includes `tests/*.py` and `pytest.ini`

### Fixed
- **Empty code initialization**: Fixed missing attributes in truly empty codes
  - `unique_sizes` and `indices_by_size` now properly initialized
  - `min_size` set to `float('inf')` for empty codes instead of 0
  
- **`has_empty_set()` false positives**: Added `n_words > 0` check to prevent false positives when code is completely empty

- **Circular import in examples.py**: Changed from absolute import `from combinatorial_codes import CombinatorialCode` to relative import `from .codes import CombinatorialCode`

### Enhanced
- **`has_empty_set()` method**: Improved logic to handle edge cases correctly
- **Test coverage**: Added comprehensive tests for `add_empty_word()` method with 5 test cases:
  - Empty code initialization
  - Adding to existing code without empty set
  - Attempting to add when empty set already exists
  - Attribute verification after addition
  - Integration with other methods

- **README.md**: Added Installation Verification section documenting the new `verify_installation()` functionality

### Technical Details
- Fixed attribute management in empty code edge cases
- Improved error handling in post-installation verification
- Enhanced package metadata and distribution configuration
- Added comprehensive regression testing for computational correctness

### Performance
- Verification typically completes in 4-6 seconds
- C extension detection and performance validation included
- Reference computation times: ~2 seconds for intersections, ~2.5 seconds for obstructions

### Dependencies
No changes to core dependencies. Test dependencies managed via `extras_require`.

---

## Previous Versions
- [0.1.x] - Initial releases with basic combinatorial codes functionality