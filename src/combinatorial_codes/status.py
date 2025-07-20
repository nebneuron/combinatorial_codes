"""
Status checking utilities for combinatorial_codes package.
"""

def check_c_extension_status():
    """
    Check and report the status of C extensions.
    
    Returns:
        bool: True if C extensions are available, False otherwise
    """
    try:
        from . import utils
        
        print("=" * 50)
        print("COMBINATORIAL CODES - EXTENSION STATUS")
        print("=" * 50)
        
        if utils.tf is not None:
            print("✓ C extensions: ACTIVE")
            print("✓ Performance mode: HIGH")
            available_funcs = [name for name in dir(utils.tf) if not name.startswith('_')]
            print(f"✓ Available functions: {len(available_funcs)}")
            print("✓ Expected speedup: 5-10x for large computations")
            result = True
        else:
            print("⚠ C extensions: NOT AVAILABLE")
            print("⚠ Performance mode: STANDARD")
            print("⚠ Using Python/Numba fallback implementations")
            print("\nTo enable C extensions:")
            print("  • macOS: xcode-select --install")  
            print("  • Linux: sudo apt-get install python3-dev gcc")
            print("  • Windows: Microsoft Visual C++ Build Tools")
            print("  • Then: pip install --force-reinstall combinatorial_codes")
            result = False
            
        print("=" * 50)
        return result
        
    except ImportError as e:
        print(f"Error: Cannot import combinatorial_codes: {e}")
        return False

def quick_status():
    """Quick one-line status check"""
    try:
        from . import utils
        if utils.tf is not None:
            return "C extensions: ACTIVE (High Performance)"
        else:
            return "C extensions: NOT AVAILABLE (Standard Performance)"
    except ImportError:
        return "C extensions: ERROR (Import Failed)"
