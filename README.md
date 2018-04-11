# Python PNI 

Python wrapper for the *libpniio* and *libpnicore* C++ libraries. The wrapper
supports Python 2.X and 3.X.

## Installation

### Required packages

* *libpniio* and *libpnicore* version >= 1.0.0
* python setuptools
* numpy
* c++ compiler
* boost-python library
* python sphinx to build the documentation


### Install from sources

The code can be built with 

```
$ python setup.py install 
```

For those who are still running on the old interface it is maybe whise to
install this package in a custom location with something like this 

```
$ python setup.py install --prefix=<path to installation prefix>
```

To build the documentation use 

```
$ python setup.py build_sphinx
```

The resulting documentation can be found below `buil/sphinx/html` in the root
directory of the source distribution.

Finally, the package can be tested using 

```
$ python setup.py test 
```

For Python3 just replace python with python3 in the above instructions.

More information can be found at [online documentation](https://pni-libraries.github.io/python-pni/index.html).