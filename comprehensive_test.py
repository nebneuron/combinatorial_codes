#!/usr/bin/env python3

print("=== Testing C Extension Import ===")

# Test 1: Direct import
print("\n1. Testing direct import...")
try:
    import combinatorial_codes.translated_functions as tf_direct
    print(f"✓ Direct import successful")
    print(f"  Functions: {len([x for x in dir(tf_direct) if not x.startswith('_')])} available")
except ImportError as e:
    print(f"✗ Direct import failed: {e}")

# Test 2: Import through utils
print("\n2. Testing import through utils...")
try:
    import combinatorial_codes.utils as utils
    print(f"✓ Utils imported")
    print(f"  utils.tf = {utils.tf}")
    print(f"  Type: {type(utils.tf)}")
    if utils.tf is not None:
        funcs = [x for x in dir(utils.tf) if not x.startswith('_')]
        print(f"  Functions available: {len(funcs)}")
        print(f"  Sample functions: {funcs[:3]}")
    else:
        print("  ✗ utils.tf is None!")
except Exception as e:
    print(f"✗ Utils import issue: {e}")

# Test 3: Check status
print("\n3. Testing status function...")
try:
    from combinatorial_codes.status import check_c_extension_status
    result = check_c_extension_status()
    print(f"Status function result: {result}")
except Exception as e:
    print(f"✗ Status check failed: {e}")

print("\n=== End Test ===")
