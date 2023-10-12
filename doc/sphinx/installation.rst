============
Installation
============

Build requirements
==================

For the build of the package the following dependencies must be satisfied

+--------------+----------------+
| package      | version        |
+==============+================+
| libh5cpp     | >=0.5.0        |
+--------------+----------------+
| libpninexus  | >=3.0.0        |
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

Debian and Ubuntu users
=======================

As Debian and Ubuntu are closely related the installation is quite similar.
The packages are provided by a special Debian repository. To work on the
package sources you need to login as `root` user. Use :command:`su` or
:command:`sudo su` on Debian and Ubuntu respectively.
The first task is to add the GPG key of the HDRI repository to your local
keyring

.. code-block:: bash

   $ curl -s http://repos.pni-hdri.de/debian_repo.pub.gpg  | gpg --no-default-keyring --keyring gnupg-ring:/etc/apt/trusted.gpg.d/debian-hdri-repo.gpg --import
   $ chmod 644 /etc/apt/trusted.gpg.d/debian-hdri-repo.gpg

The return value of this command line should be `OK`.
In a next step you have to add new package sources to your system. For this
purpose go to :file:`/etc/apt/sources.list.d` and download the sources file.
For Debian (Bookworm) use

.. code-block:: bash

   $ wget http://repos.pni-hdri.de/bookworm-hdri.list

and for Ubuntu (Jammy)

.. code-block:: bash

   $ wget http://repos.pni-hdri.de/jammy-pni-hdri.list

Similarly, proceed for Bookworm, Bullseye, Buster, Lunar, Jammy, Focal.
Once you have downloaded the file use

.. code-block:: bash

   $ apt-get update


to update your package list and

.. code-block:: bash

   $ apt-get install python-pninexus

or

.. code-block:: bash

   $ apt-get install python3-pninexus

to install the the PNI/NeXus package for python or python3, respectively.

Manylinux wheels from PyPI
==========================

To install pninexus in a conda or python virtual environment manylinux pip wheels can be used, e.g.

.. code-block:: console

   $ python3 -m venv myvenv
   $ . myvenv/bin/activate
   
   $ pip install pip --upgrade
   $ pip install numpy
   
   $ pip install pninexus -U wheel

For some versions of wheels numpy needs to be installed in advance.



More information can be found at `online
documentation <https://pni-libraries.github.io/python-pninexus/index.html>`__.

Changes for a specific version of libpninexus can be found at
`CHANGELOG <https://github.com/pni-libraries/python-pninexus/blob/develop/CHANGELOG.md>`__.
