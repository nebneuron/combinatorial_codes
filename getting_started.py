#!/usr/bin/env python3
"""
Getting Started with combinatorial_codes

This script demonstrates basic usage of the combinatorial_codes package
and runs the key validation tests requested.
"""

def main():
    print("🚀 Getting Started with combinatorial_codes")
    print("=" * 50)
    
    try:
        from combinatorial_codes import CombinatorialCode, example_code, check_c_extension_status
        import numpy as np
        
        # Check package status
        print("📦 Package Status:")
        has_c_extensions = check_c_extension_status()
        print()
        
        # Example 1: Create a simple code
        print("📝 Example 1: Creating a simple combinatorial code")
        print("-" * 40)
        simple_vectors = [[], [1], [2], [1, 2]]
        simple_code = CombinatorialCode(simple_vectors)
        print(f"Created code with {len(simple_code.words)} words")
        print(f"Has empty set: {simple_code.has_empty_set()}")
        print()
        
        # Example 2: Using example codes
        print("📝 Example 2: Using built-in example codes")
        print("-" * 40)
        eyes_code = example_code('eyes')
        print(f"'Eyes' code: {len(eyes_code.words)} words")
        
        # Example 3: The requested validation tests
        print("📝 Example 3: Validation Tests (Your Requirements)")
        print("-" * 40)
        milo_code = example_code("example by Milo")
        
        # Test 1: Simplicial violators
        violators = milo_code.simplicial_violators()
        expected_violators = np.array([4, 5, 8, 12, 16, 20, 32, 64, 65, 144, 256, 528, 1040, 1536, 2048, 2052, 2056, 2320, 18432], dtype=np.uint64)
        violators_match = np.array_equal(np.sort(violators), np.sort(expected_violators))
        
        print(f"Simplicial violators test: {'✅ PASS' if violators_match else '❌ FAIL'}")
        print(f"  Found {len(violators)} violators")
        
        # Test 2: Obstructions
        is_complete, num_obstructions = milo_code.Obstructions()
        expected_obstructions = (False, 17)
        obstructions_match = (is_complete, num_obstructions) == expected_obstructions
        
        print(f"Obstructions test: {'✅ PASS' if obstructions_match else '❌ FAIL'}")
        print(f"  Result: ({is_complete}, {num_obstructions})")
        
        # Summary
        print()
        print("🎉 Summary:")
        print("-" * 40)
        if violators_match and obstructions_match:
            print("✅ All validation tests PASSED!")
            print("✅ Package is working correctly!")
        else:
            print("❌ Some tests failed - please check installation")
            
        print("\n📚 Next Steps:")
        print("- Read the documentation in README.md")
        print("- Run full tests with: python run_tests.py")
        print("- Explore other example codes: 'open not closed', 'closed not open'")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n🔧 Try installing the package first:")
        print("   pip install -e .")
        return 1
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
