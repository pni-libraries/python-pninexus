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

The resulting documentation can be found below `build/sphinx/html` in the root
directory of the source distribution.

Finally, the package can be tested using

```
    $ python setup.py test
```

For Python3 just replace python with python3 in the above instructions.


### Debian and Ubuntu packages

Debian  `bullseye`, `buster`, `stretch` or Ubuntu  `impish`, `focal`, `bionic` packages can be found in the HDRI repository.

To install the debian packages, add the PGP repository key

```
    $ sudo su
    $ curl -s http://repos.pni-hdri.de/debian_repo.pub.gpg  | gpg --no-default-keyring --keyring gnupg-ring:/etc/apt/trusted.gpg.d/debian-hdri-repo.gpg --import
    $ chmod 644 /etc/apt/trusted.gpg.d/debian-hdri-repo.gpg
```

and then download the corresponding source list, e.g.
for `bullseye`

```
    $ cd /etc/apt/sources.list.d
    $ wget http://repos.pni-hdri.de/bullseye-pni-hdri.list
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
