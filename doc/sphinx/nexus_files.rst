
Working with Nexus files
========================


The most fundamental thing you may want to do is to create, open, and close
files. File creation and closing could look like this 

.. code-block:: python
    
    import pni.io.nx.h5 as nexus

    f = nexus.create_file("test.nxs")
    #... do some work here
    f.close()

If the file you try to create already exists an exception will be thrown. To
override this behavior the :py:func:`create_file` function has a boolean 
keyword argument `overwrite`

.. code-block:: python

    f = nexus.create_file("text.nxs",overwrite=True)

When using ``overwrite=True`` an already existing file will be overwritten. 
Existing files can be opened using the :py:func:`open_file` function

.. code-block:: python
    
    f = nexus.open_file("test.nxs")

which by default opens an existing file in read-only mode. Use

.. code-block:: python

    f = nexus.open_file("test.nxs",readonly=False)

to open the file for reading and writing.  The file object returned by
:py:func:`create_file` and :py:func:`open_file` provides only very limited
functionality.  In order to do some useful work you have to obtain the root
node of the Nexus tree (in HDF5 this is the root group of the file) with

.. code-block:: python

    root = f.root()

The :py:func:`create_file` function creates a simple single file into which all
data will be stored. This can causes some problems when the files become large 
(several hundred Gigabytes) and the data should be archived in a tape library
or if the data should be transfered via a network. The HDF5 library offers a
possiblity to split files into smaller portions. `libpniio` provides a 
simple interface to this functionality via the :py:func:`create_files` function
(note the usage of the plural `files` in the functions name).
The :py:func:`create_files` function works quite similar to its single file 
counterpart :py:func:`create_file` with to exceptions

* it has an additional keyword argument `split_size` which determines the 
  size of the individual files
* the filename must be a C-like format string encoding a running file index. 

In code this could look like this

.. code-block:: python

    f = nexus.create_files("test.%05i.nxs",
                           overwrite=True,
                           split_size=30*1024**3)


The file will be split into subfiles of 30 GByte size. After filling data into
the file we would get files like 

.. code-block:: bash

    $ ls 
      test.00001.nxs
      test.00002.nxs
      test.00003.nxs

It is important to note that the files cannot be used invidually. 

In order to
access the data all of the must be available and should be valid. 
Opening such a `distributed` file straight forward, just pass the C-format
string as a filename to the :py:func:`open_file` function

.. code-block:: python

    f = nexus.open_file('test.%05i.nxs')

No additional action has to be taken by the user. The splitted file behaves the
same as the single flie created by :py:func:`create_file`.
