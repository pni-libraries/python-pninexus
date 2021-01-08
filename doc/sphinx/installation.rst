============
Installation
============

Build requirements
==================

For the build of the package the following dependencies must be satisfied

+--------------+----------------+
| package      | version        |
+==============+================+
| libh5cpp     | >=0.4.0        |
+--------------+----------------+
| libpnicore   | >=1.1.0        |
+--------------+----------------+
| libpniio     | >=1.2.0        |
+--------------+----------------+
| libboost     | >=1.60.0       |
+--------------+----------------+
| numpy        |                |
+--------------+----------------+
| gcc compiler | >=4.7          |
+--------------+----------------+
| python       | >=2.7 or >=3.4 |
+--------------+----------------+


Building the package
====================

The wrapper package follows the standard Python build and installation
protocoll using Python setuptools. 

.. code-block:: bash

    $ python setup.py build
    $ python setup.py install

or for Python3

.. code-block:: bash

    $ python3 setup.py build
    $ python3 setup.py install

In order to build the Sphinx documentation use 

.. code-block:: bash

    $ python3 setup.py build_sphinx 


