#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, Extension


def get_extensions():
    import pybind11
    import numpy
    from Cython.Build import cythonize
    exts = [
        Extension(
            "bls._impl",
            sources=[
                os.path.join("bls", "bls.c"),
                os.path.join("bls", "_impl.pyx"),
            ],
            include_dirs=[numpy.get_include()],
        ),
        Extension(
            "bls.grid_search",
            [os.path.join("bls", "grid_search.cc")],
            include_dirs=[
                pybind11.get_include(False),
                pybind11.get_include(True),
                numpy.get_include(),
            ],
            language="c++",
            extra_compile_args=["-O2", "-std=c++14", "-stdlib=libc++"],
        ),
    ]

    return cythonize(exts)


setup(
    name="bls.py",
    version="0.1.2",
    author="Daniel Foreman-Mackey & Ze Vinicius",
    author_email="foreman.mackey@gmail.com",
    url="https://github.com/dfm/bls.py",
    description="A reference implementation of box least squares in Python",
    long_description=open("README.rst").read(),
    license="BSD",
    packages=["bls"],
    ext_modules=get_extensions(),
    setup_requires=["numpy", "cython", "pybind11"],
)
