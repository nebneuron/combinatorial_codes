"""
Post-installation verification for combinatorial_codes package.

This module runs regression tests after package installation to verify
that everything is working correctly.
"""

import sys
import time
import traceback
from typing import Tuple, List

def run_regression_tests() -> Tuple[bool, List[str]]:
    """
    Run regression tests and return success status and messages.
    
    Returns:
        Tuple[bool, List[str]]: (success, messages)
    """
    messages = []
    success = True
    
    try:
        messages.append("🧪 Running post-installation regression tests...")
        
        # Test 1: Basic imports
        messages.append("  ⏳ Testing package imports...")
        from combinatorial_codes import CombinatorialCode, example_code
        from combinatorial_codes.utils import intersections_via_cliques
        from combinatorial_codes.correct_examples import (
            RANDOM_EXAMPLE_32_INTERSECTIONS,
            RANDOM_EXAMPLE_32_OBSTRUCTIONS
        )
        import numpy as np
        messages.append("  ✅ Package imports successful")
        
        # Test 2: C extension status
        messages.append("  ⏳ Checking C extension status...")
        try:
            from combinatorial_codes import check_c_extension_status, quick_status
            status = quick_status()
            messages.append(f"  ✅ Extension status: {status}")
        except Exception as e:
            messages.append(f"  ⚠️  Extension status check failed: {e}")
        
        # Test 3: Example code loading
        messages.append("  ⏳ Testing example code loading...")
        C = example_code("random example 32")
        messages.append(f"  ✅ Loaded random example 32: {C.n_words} words, {len(C.maximal_words)} maximal")
        
        # Test 4: Intersections computation
        messages.append("  ⏳ Testing intersections computation...")
        start_time = time.time()
        intersections = intersections_via_cliques(C.maximal_words)
        intersections_time = time.time() - start_time
        
        # Verify against reference
        intersections_match = np.array_equal(intersections, RANDOM_EXAMPLE_32_INTERSECTIONS)
        if intersections_match:
            messages.append(f"  ✅ Intersections correct: {len(intersections)} values ({intersections_time:.2f}s)")
        else:
            messages.append(f"  ❌ Intersections mismatch: expected {len(RANDOM_EXAMPLE_32_INTERSECTIONS)}, got {len(intersections)}")
            success = False
            
        # Test 5: Obstructions computation
        messages.append("  ⏳ Testing obstructions computation...")
        start_time = time.time()
        is_complete, num_obstructions = C.Obstructions()
        obstructions_time = time.time() - start_time
        
        # Verify against reference
        expected_is_complete, expected_num_obstructions = RANDOM_EXAMPLE_32_OBSTRUCTIONS
        obstructions_match = (is_complete == expected_is_complete and 
                            num_obstructions == expected_num_obstructions)
        
        if obstructions_match:
            messages.append(f"  ✅ Obstructions correct: {num_obstructions} obstructions ({obstructions_time:.2f}s)")
        else:
            messages.append(f"  ❌ Obstructions mismatch: expected {RANDOM_EXAMPLE_32_OBSTRUCTIONS}, got ({is_complete}, {num_obstructions})")
            success = False
            
        # Test 6: Additional functionality
        messages.append("  ⏳ Testing additional functionality...")
        try:
            # Test add_empty_word method
            test_code = CombinatorialCode([[1], [2]])
            result = test_code.add_empty_word()
            if result and test_code.has_empty_set():
                messages.append("  ✅ add_empty_word method working")
            else:
                messages.append("  ❌ add_empty_word method failed")
                success = False
        except Exception as e:
            messages.append(f"  ❌ Additional functionality test failed: {e}")
            success = False
        
        # Final result
        if success:
            messages.append("🎉 All regression tests passed! Package installation verified.")
        else:
            messages.append("💥 Some regression tests failed. Please check your installation.")
            
    except Exception as e:
        success = False
        messages.append(f"❌ Regression test failure: {e}")
        messages.append("📋 Full traceback:")
        messages.extend(f"   {line}" for line in traceback.format_exc().splitlines())
        
    return success, messages

def verify_installation():
    """
    Run installation verification and print results.
    Called during package installation.
    """
    print("\n" + "="*60)
    print("COMBINATORIAL CODES - POST-INSTALLATION VERIFICATION")
    print("="*60)
    
    success, messages = run_regression_tests()
    
    for message in messages:
        print(message)
    
    print("="*60)
    if success:
        print("✅ Installation verification completed successfully!")
        print("The combinatorial_codes package is ready to use.")
    else:
        print("⚠️  Installation verification detected issues.")
        print("The package may still be usable, but some features might not work correctly.")
        print("Please check the messages above or contact support.")
    print("="*60 + "\n")
    
    return success

if __name__ == "__main__":
    # Allow running verification manually
    success = verify_installation()
    sys.exit(0 if success else 1)