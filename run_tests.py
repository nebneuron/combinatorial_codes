"""
Test runner script for combinatorial_codes package.

This script provides a convenient way to run different types of tests.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and print its description."""
    print(f"\n{'='*50}")
    print(f"🧪 {description}")
    print(f"{'='*50}")
    
    result = subprocess.run(command, shell=True, cwd=os.path.dirname(__file__))
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully!")
    else:
        print(f"❌ {description} failed with code {result.returncode}")
        
    return result.returncode

def main():
    """Main test runner function."""
    
    # Get the virtual environment python path
    venv_python = "/Users/vui1/Documents/GitHub/combinatorial_codes/.venv/bin/python"
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("""
Usage: python run_tests.py [option]

Options:
  --help          Show this help message
  --all           Run all tests (default)
  --milo          Run only the Milo example tests
  --fast          Run quick tests only
  --coverage      Run tests with coverage report

Examples:
  python run_tests.py                # Run all tests
  python run_tests.py --milo         # Test Milo example only
  python run_tests.py --coverage     # Generate coverage report
        """)
        return
        
    option = sys.argv[1] if len(sys.argv) > 1 else "--all"
    
    if option == "--milo":
        # Run only the Milo example tests
        exit_code = run_command(
            f"{venv_python} -m pytest tests/test_combinatorial_codes.py::TestExampleCodes::test_milo_example_simplicial_violators tests/test_combinatorial_codes.py::TestExampleCodes::test_milo_example_obstructions -v",
            "Running Milo Example Tests"
        )
        
    elif option == "--fast":
        # Run tests excluding slow ones
        exit_code = run_command(
            f"{venv_python} -m pytest tests/ -v -m 'not slow'",
            "Running Fast Tests"
        )
        
    elif option == "--coverage":
        # Run tests with coverage
        exit_code = run_command(
            f"{venv_python} -m pytest tests/ --cov=combinatorial_codes --cov-report=html --cov-report=term",
            "Running Tests with Coverage"
        )
        
    else:  # --all or default
        # Run all tests
        exit_code = run_command(
            f"{venv_python} -m pytest tests/ -v",
            "Running All Tests"
        )
        
    # Summary
    print(f"\n{'='*50}")
    if exit_code == 0:
        print("🎉 All selected tests passed!")
    else:
        print("💥 Some tests failed. Check output above.")
    print(f"{'='*50}\n")
        
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
