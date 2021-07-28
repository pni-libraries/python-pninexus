# Python bindings for PNI/NeXus and h5cpp

Python wrapper for the [h5cpp](https://github.com/ess-dmsc/h5cpp)  and [libpninexus](https://github.com/pni-libraries/libpninexus) C++ libraries.
The wrapper supports Python 2.X and 3.X.

## Installation

### Required packages

* *h5cpp*  >= 0.4.0
* *libpninexus* >= 2.0.0
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


### Debian and Ubuntu packages

Debian `buster`, `stretch` and `bullseye` or Ubuntu  `groovy` `focal`, `bionic` packages can be found in the HDRI repository.

To install the debian packages, add the PGP repository key

```
    $ sudo su
    $ wget -q -O - http://repos.pni-hdri.de/debian_repo.pub.gpg | apt-key add -
```

and then download the corresponding source list, e.g.
for `buster`

```
    $ cd /etc/apt/sources.list.d
    $ wget http://repos.pni-hdri.de/buster-pni-hdri.list
```

or `focal`

```
    $ cd /etc/apt/sources.list.d
    $ wget http://repos.pni-hdri.de/focal-pni-hdri.list
```
respectively.

Finally,

```
    $ apt-get update
    $ apt-get install python-pninexus
```

or

```
    $ apt-get update
    $ apt-get install python3-pninexus
```

for python3.

More information can be found at [online documentation](https://pni-libraries.github.io/python-pninexus/index.html).

Changes for a specific version of libpninexus can be found
at [CHANGELOG](https://github.com/pni-libraries/python-pninexus/blob/develop/CHANGELOG.md).
