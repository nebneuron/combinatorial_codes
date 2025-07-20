# C Extension Compilation

The `combinatorial_codes` package includes optional C extensions that provide significant performance improvements for computationally intensive operations. These extensions are automatically compiled during package installation.

## Installation with Status Feedback

When you install the package using `pip install`, you'll see explicit feedback about C extension compilation:

```bash
pip install -e .
```

During installation, you'll see messages like:
```
============================================================
Building combinatorial_codes C extensions...
============================================================
✓ SUCCESS: C extensions compiled successfully!
  Performance-optimized functions are now available.
  Expected speedup: 5-10x for computationally intensive operations.
============================================================
```

## Checking Installation Status

After installation, you can easily check if C extensions are working:

### Quick Status Check
```python
import combinatorial_codes as cc
print(cc.quick_status())
# Output: "C extensions: ACTIVE (High Performance)"
```

### Detailed Status Check
```python
import combinatorial_codes as cc
cc.check_c_extension_status()
```

This will show:
```
==================================================
COMBINATORIAL CODES - EXTENSION STATUS
==================================================
✓ C extensions: ACTIVE
✓ Performance mode: HIGH
✓ Available functions: 6
✓ Expected speedup: 5-10x for large computations
==================================================
```

### Comprehensive Test
```bash
python test_c_extension.py
```

This runs a complete test of both C extension availability and core functionality.

## Automatic Compilation

The C extensions will be automatically compiled if:

1. A C compiler is available on your system
2. NumPy development headers are accessible
3. Python development headers are available

## Fallback Behavior

If the C extensions fail to compile, you'll see a warning message during installation:
```
⚠ WARNING: C extension compilation failed: [error details]
  The package will still work using Python/Numba implementations.
  Performance will be slower but functionality remains intact.
```

The package will still work using Python/Numba implementations.

## Platform-Specific Notes

### macOS
- Requires Xcode command line tools: `xcode-select --install`
- Works with both Intel and Apple Silicon

### Linux
- Requires `gcc` or compatible C compiler
- Install development packages: `sudo apt-get install python3-dev` (Ubuntu/Debian)

### Windows
- Requires Microsoft Visual C++ Build Tools
- Or use conda which typically includes necessary compilers

## Performance

The C extensions provide significant speedups for:
- `intersections_inside_a_clique`: ~5-10x faster
- `intersection_of_codewords_from_bits`: ~3-5x faster
- `lattice_slice` operations: ~2-3x faster

For large combinatorial codes with many maximal words, these performance improvements can reduce computation time from hours to minutes.
