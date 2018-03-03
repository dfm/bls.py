#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, Extension


def get_extensions():
    import numpy
    from Cython.Build import cythonize
    ext = Extension(
        "transit_periodogram._impl",
        sources=[
            os.path.join("transit_periodogram", "transit_periodogram.c"),
            os.path.join("transit_periodogram", "_impl.pyx"),
        ],
        include_dirs=[numpy.get_include()],
    )
    return cythonize([ext])


setup(
    name="transit_periodogram",
    version="0.1.0",
    author="Daniel Foreman-Mackey & Ze Vinicius",
    author_email="foreman.mackey@gmail.com",
    url="https://github.com/dfm/astropy-transit-periodogram",
    license="BSD",
    packages=["transit_periodogram"],
    ext_modules=get_extensions(),
    setup_require=["numpy", "cython"],
)
