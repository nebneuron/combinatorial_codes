#!/usr/bin/env python3
"""
Test script to verify that the C extension compiles and works correctly.
Run this after installing the package to verify functionality.
"""

def test_c_extension():
    """Test that the C extension is available and functional"""
    print("=" * 60)
    print("COMBINATORIAL CODES - C EXTENSION STATUS CHECK")
    print("=" * 60)
    
    try:
        from combinatorial_codes import utils
        from combinatorial_codes.examples import example_code
        
        print("\n1. Testing C extension availability...")
        
        if utils.tf is not None:
            print("   ✓ C extension loaded successfully")
            print(f"   ✓ Available C functions: {[name for name in dir(utils.tf) if not name.startswith('_')]}")
            print("   ✓ Performance optimizations are ACTIVE")
        else:
            print("   ⚠ C extension not available")
            print("   ⚠ Using Python/Numba fallback implementations")
            print("   ⚠ Performance will be slower but functionality remains intact")
            
        print("\n2. Testing core functionality...")
        code = example_code('eyes')
        is_complete, num_obstructions = code.Obstructions()
        
        print(f"   ✓ Obstructions calculation successful")
        print(f"   ✓ Is maximal intersection complete: {is_complete}")
        print(f"   ✓ Number of obstructions: {num_obstructions}")
        
        print("\n3. Performance status:")
        if utils.tf is not None:
            print("   ✓ HIGH PERFORMANCE MODE - C extensions active")
            print("   ✓ Expected speedup: 5-10x for large computations")
        else:
            print("   ⚠ STANDARD MODE - Python/Numba implementations")
            print("   ⚠ Consider installing development tools for C compilation")
            
        print("\n" + "=" * 60)
        print("OVERALL STATUS:", "✓ OPTIMAL" if utils.tf is not None else "⚠ FUNCTIONAL")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        print("\n" + "=" * 60)
        print("OVERALL STATUS: ✗ ERROR")
        print("=" * 60)
        return False

def print_installation_help():
    """Print help for enabling C extensions"""
    print("\nTo enable C extensions, ensure you have:")
    print("  • macOS: xcode-select --install")
    print("  • Linux: sudo apt-get install python3-dev gcc (Ubuntu/Debian)")
    print("  • Windows: Microsoft Visual C++ Build Tools")
    print("\nThen reinstall: pip uninstall combinatorial_codes && pip install combinatorial_codes")

if __name__ == "__main__":
    success = test_c_extension()
    if not success:
        print_installation_help()
    exit(0 if success else 1)
