# Python PNI 

Python wrapper for the *libpniio* and *libpnicore* C++ libraries. 

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

