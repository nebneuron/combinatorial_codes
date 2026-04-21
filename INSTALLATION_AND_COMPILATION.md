# Installation and Compilation Guide

This guide collects the installation, verification, and C extension build details that were previously embedded in the main README.

This package has been tested only on Linux and macOS.

## Installation

### Quick Install

To install just the package for use:

```bash
pip install -e .
```

### Install with Testing Support

To install the package and run tests:

```bash
pip install -e ".[test]"
python run_tests.py
```

### One-Command Install and Test

For convenience, you can install and test in one step:

```bash
python install_and_test.py
```

### Quick Start Demo

To verify installation and see basic usage:

```bash
python getting_started.py
```

## Installation Verification

The package includes automatic post-installation verification that runs regression tests during `pip install`. You can also verify manually:

```python
from combinatorial_codes import verify_installation

verify_installation()
```

This verification covers:

- package imports and C extension detection
- computational correctness using reference examples
- performance benchmarking of key operations
- verification of `add_empty_word()`

The verification typically takes 4-6 seconds.

## C Extension Compilation

This package includes optional C extensions for performance-critical operations. The package still works without them, but large computations will be slower.

### System Requirements

Linux or macOS:

```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel

# macOS
xcode-select --install
```

### Build Process

The C extensions are compiled automatically during installation.

### Check C Extension Status

If compilation fails, the package falls back to Python or Numba implementations. You can inspect the status with:

```python
from combinatorial_codes import check_c_extension_status

check_c_extension_status()
```

### Common Issues

`No module named 'combinatorial_codes.translated_functions'`

- The C extension did not compile.
- Install the required build tools for your platform.
- Make sure Python development headers are available.

`clang: error: unsupported option`

- Update the Xcode command line tools with `xcode-select --install`.
- Confirm that your Python version is compatible with the local toolchain.

Cross-platform moves:

- C extensions are built for a specific Python version and architecture.
- If you move the repo between systems, reinstall the package:

```bash
pip uninstall combinatorial_codes
pip install -e .
```

### Performance Impact

- With C extensions: roughly 5-10x faster on large codes
- Without C extensions: slower, but still functional through Numba JIT compilation

To force a rebuild of the extensions:

```bash
pip install -e . --force-reinstall --no-deps
```

## Testing

Run all tests:

```bash
python run_tests.py
pytest tests/ -v
```

Run only the key example tests:

```bash
python run_tests.py --milo
```
