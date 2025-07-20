#!/usr/bin/env python3
"""
Convenience script for installing combinatorial_codes with automatic testing.

Usage:
    python install_and_test.py
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    """Install package and run tests"""
    print("🚀 Installing combinatorial_codes with automatic testing")
    print("=" * 60)
    
    # Get Python executable
    python_exe = sys.executable
    
    # Step 1: Install package in development mode
    if not run_command(f"{python_exe} -m pip install -e .", 
                      "Installing package in development mode"):
        print("❌ Installation failed!")
        return 1
    
    # Step 2: Install test dependencies
    if not run_command(f"{python_exe} -m pip install -e '.[test]'", 
                      "Installing test dependencies"):
        print("❌ Test dependency installation failed!")
        return 1
    
    # Step 3: Run tests
    if not run_command(f"{python_exe} run_tests.py", 
                      "Running test suite"):
        print("❌ Tests failed!")
        print("\n⚠️  Package installed successfully, but some tests failed.")
        print("   The package should still work for basic functionality.")
        return 1
    
    print("\n🎉 SUCCESS!")
    print("=" * 60)
    print("✅ Package installed successfully")
    print("✅ All tests passed")
    print("✅ Ready to use!")
    
    # Show quick usage example
    print("\n📖 Quick usage example:")
    print("python -c \"from combinatorial_codes import example_code; print(example_code('eyes'))\"")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
