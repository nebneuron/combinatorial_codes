# setup.py
from setuptools import Extension, setup
import numpy as np

setup(
    name="translated_functions",
    version="0.1.0",
    ext_modules=[
        Extension(
            "translated_functions",
            ["translated_functions.c"],
            include_dirs=[np.get_include()],
        )
    ],
)