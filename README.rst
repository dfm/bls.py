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

Usage
-----

See ``tutorial.ipynb`` for a demonstration of how to use the code.
