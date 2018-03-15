Transit Periodogram
===================

A reference implementation of the transit periodogram for AstroPy.
This will eventually be submitted as a pull request to the core AstroPy package, but
we're making it available here for now to make it easy to use right away.

Prerequisites
-------------

To install and run this package, you will need to have NumPy, Cython, and AstroPy installed.
The recommended installation method is:

.. code-block:: bash
 
    conda install numpy cython astropy

Installation
------------

To install, clone this repository and build the extension as follows:

.. code-block:: bash
    
    git clone https://github.com/dfm/astropy-transit-periodogram.git
    cd astropy-transit-periodogram
    python setup.py install
    
or, install using ``pip``:

.. code-block:: bash

    pip install https://github.com/dfm/astropy-transit-periodogram/archive/master.zip
    
**OpenMP support**: This algorithm can optionally be parallelized using OpenMP.
To enable this feature, you must compile with a compiler that supports OpenMP and the
relevant flags. On macOS, this can be achieved by installing a recent ``llvm``:

.. code-block:: bash

    brew install llvm
 
and then building using the following flags:

.. code-block:: bash

    CC=/usr/local/opt/llvm/bin/clang \
     LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib -liomp5" \
     CFLAGS="-I/usr/local/opt/llvm/include -fopenmp" \
     python setup.py install
    
On other platforms, a command like the following might be sufficient:

.. code-block:: bash

    CFLAGS="-lgomp -fopenmp" python setup.py install

Usage
-----

See ``tutorial.ipynb`` for a demonstration of how to use the code.
