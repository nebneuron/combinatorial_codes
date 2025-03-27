from setuptools import setup, find_packages

setup(
    name="combinatorial_codes",
    version="0.1.0",
    description="A Python package for manipulating combinatorial codes",
    long_description="",
    author="Vladimir Itskov",
    author_email="vladimir.itskov@psu.edu",
    url="https://github.com/nebneuron/combinatorial_codes",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "numba>=0.57.0",  # Adjust version as needed
        "numpy",  # Adjust version as needed
        "gudhi>=3.11.0",  # Adjust version as needed
    ],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

