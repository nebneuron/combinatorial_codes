from setuptools import setup, find_packages

setup(
    name="combinaorial_codes",
    version="0.1.0",
    description="A Python package for manipulating combinatorial codes",
    long_description="",
    author="Vladimir Itskov",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/my_package",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "numba>=0.57.0",  # Adjust version as needed
        "numpy>=2.1.3",  # Adjust version as needed
    ],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

