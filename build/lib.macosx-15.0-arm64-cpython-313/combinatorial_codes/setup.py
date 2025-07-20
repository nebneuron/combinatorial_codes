# setup.py - Standalone C extension build (legacy)
# NOTE: This file is for standalone C extension building only.
# The main package uses the setup.py in the root directory.

from setuptools import Extension, setup
import numpy as np

setup(
    name="translated_functions",
    version="0.2.0",  # Updated to match package version
    ext_modules=[
        Extension(
            "translated_functions",
            ["translated_functions.c"],
            include_dirs=[np.get_include()],
        )
    ],
)