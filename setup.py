from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
import numpy
import sys
import platform
import subprocess

class CustomBuildExt(build_ext):
    """Custom build_ext command that provides user feedback and handles cross-platform builds"""
    
    def build_extensions(self):
        print("\n" + "="*60)
        print("Building combinatorial_codes C extensions...")
        print(f"Platform: {platform.system()} {platform.machine()}")
        print(f"Python: {sys.version}")
        print("="*60)
        
        # Platform-specific compiler flags
        for ext in self.extensions:
            if platform.system() == 'Windows':
                # MSVC flags
                ext.extra_compile_args = ['/O2', '/favor:INTEL64']
            elif platform.system() == 'Darwin':
                # macOS clang flags
                ext.extra_compile_args = ['-O3', '-ffast-math', '-march=native']
                # Handle different architectures
                if platform.machine() == 'arm64':
                    ext.extra_compile_args.append('-mcpu=apple-m1')
            else:
                # Linux gcc flags
                ext.extra_compile_args = ['-O3', '-ffast-math', '-march=native']
        
        try:
            super().build_extensions()
            print("\n✓ SUCCESS: C extensions compiled successfully!")
            print("  Performance-optimized functions are now available.")
            print("  Expected speedup: 5-10x for computationally intensive operations.")
        except Exception as e:
            print(f"\n⚠ WARNING: C extension compilation failed: {e}")
            print("  The package will still work using Python/Numba implementations.")
            print("  Performance will be slower but functionality remains intact.")
            print("\n  To enable C extensions, ensure you have:")
            if platform.system() == 'Windows':
                print("  - Microsoft Visual C++ Build Tools")
            else:
                print("  - A C compiler (gcc, clang)")
            print("  - Python development headers")
            print("  - NumPy development headers")
            # Don't re-raise the exception - allow installation to continue
            
        print("="*60 + "\n")

class CustomInstall(install):
    """Custom install command that runs post-installation verification"""
    
    def run(self):
        # Run the normal installation
        super().run()
        
        # Run post-installation verification
        try:
            # Import and run verification after installation
            print("\n" + "="*60)
            print("RUNNING POST-INSTALLATION VERIFICATION...")
            print("="*60)
            
            # We need to use subprocess because the package might not be
            # importable in the current process during installation
            result = subprocess.run([
                sys.executable, "-c",
                """
import sys
import os
# Try to import and run verification
try:
    from combinatorial_codes.install_verification import verify_installation
    verify_installation()
except ImportError:
    print('⚠️  Could not run post-installation verification.')
    print('   Package may still be functional. Try running manually:')
    print('   python -c "from combinatorial_codes.install_verification import verify_installation; verify_installation()"')
except Exception as e:
    print(f'⚠️  Post-installation verification encountered an error: {e}')
    print('   Package may still be functional.')
                """
            ], capture_output=False, text=True)
            
        except Exception as e:
            print(f"\n⚠️  Could not run post-installation verification: {e}")
            print("The package should still be functional.")
            print("You can manually run verification with:")
            print("python -c \"from combinatorial_codes.install_verification import verify_installation; verify_installation()\"")

# Define the C extension with cross-platform handling
def get_ext_modules():
    try:
        # Base configuration
        include_dirs = [numpy.get_include()]
        extra_compile_args = []
        extra_link_args = []
        
        # Platform-specific initial setup (refined in CustomBuildExt)
        if platform.system() == 'Windows':
            extra_compile_args = ['/O2']
        else:
            extra_compile_args = ['-O3', '-ffast-math']
        
        translated_functions_ext = Extension(
            'combinatorial_codes.translated_functions',
            sources=['src/combinatorial_codes/translated_functions.c'],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
        return [translated_functions_ext]
    except Exception as e:
        print(f"Warning: Could not set up C extension: {e}")
        print("The package will still work using Python/Numba implementations.")
        return []

setup(
    name="combinatorial_codes",
    version="0.2",
    description="A Python package for manipulating combinatorial codes",
    long_description="",
    author="Vladimir Itskov",
    author_email="vladimir.itskov@psu.edu",
    url="https://github.com/nebneuron/combinatorial_codes",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=get_ext_modules(),
    cmdclass={'build_ext': CustomBuildExt, 'install': CustomInstall},
    python_requires=">=3.11",
    install_requires=[
        "numba>=0.57.0",  # Adjust version as needed
        "numpy",  # Adjust version as needed
        "gudhi>=3.11.0",  # Adjust version as needed
	"networkx>=3.0",
    ],
    extras_require={
        "test": [
            "pytest>=6.0",
            "pytest-cov",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["tests/*.py", "pytest.ini"],
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

