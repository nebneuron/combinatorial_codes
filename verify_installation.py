#!/usr/bin/env python3
"""
Installation verification script for combinatorial_codes package.
This script checks if the package is properly installed and provides
information about C extension availability.
"""

import sys
import platform

def check_installation():
    print("="*60)
    print("Combinatorial Codes - Installation Verification")
    print("="*60)
    
    # System info
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Try to import the package
    try:
        import combinatorial_codes
        print(f"✓ Package imported successfully (version {combinatorial_codes.__version__})")
    except ImportError as e:
        print(f"✗ Failed to import package: {e}")
        return False
    
    # Check C extension status
    try:
        from combinatorial_codes import check_c_extension_status
        print("\nC Extension Status:")
        check_c_extension_status()
    except Exception as e:
        print(f"⚠ Could not check C extension status: {e}")
    
    # Quick functionality test
    try:
        from combinatorial_codes import CombinatorialCode, example_code
        print("\nQuick functionality test:")
        
        # Test with a simple code
        test_code = CombinatorialCode([[],[1],[2],[1,2]])
        print(f"✓ Created test code with {test_code.n_words} codewords")
        
        # Test example code
        ec = example_code()
        print(f"✓ Example code loaded with {ec.n_words} codewords")
        
        print("✓ Basic functionality verified")
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("✓ Installation verification completed successfully!")
    print("The package is ready to use.")
    print("="*60)
    return True

if __name__ == "__main__":
    success = check_installation()
    sys.exit(0 if success else 1)
