Working with fields
===================

Nexus fields are the data holding instance in a Nexus files. On can imagine a
field like a multidimensional array stored on disk. In fact the field class 
used in this python package is designed to behave a little like the famous numpy
arrays. So they should be rather easy to use. 

The simplest way to create a field would look like this

.. code-block:: python

    name = entry.create_field("name",type="str")

which creates a 1D field with a single element of type string. The first
argument is mandatory as it describes the name of the field.  Like with the
``create_group`` method ``create_field`` can only create a direct child of its
parent group. The method accepts three more keyword arguments: ``shape``,
``chunk``, and ``filter``.  ``shape`` determines the shape of the field after
it has been created.  ``chunk`` is the chunk shape used for the field and
finally ``filter`` can be a filter instance used to compress the data during
writing.  It is important to note that the ``create_field`` method will always
set a chunk size, even when none is provided by the user. This is a requirement
in order to get fields which can be grown (see next section).  In the next
example we will use all the keyword arguments

.. code-block:: python

    import pni.io.nx.h5 as nx

    file = nx.open_file("data.nxs",readonly=False)
    root = file.root();

    path = "/:NXentry/:NXinstrument/:NXdetector"
    detector = get_object(root,path)

    #create a deflate filter with compression level 5
    filter = nx.deflate_filter(5)

    detector.create_field("data",
                          type="uint16",
                          shape=(0,1024,2048),  #set the initial field size
                          chunk=(1,1024,2048),  #set the chunk size
                          filter = filter)

                          
There are two things which are important here. Please note that the first
element in the shape of the field is $0$. This means its total size will be $0$
and no data can be stored. This is a quite reasonable approach if one does not 
know how many elements (lets say image frames) will be stored in the field. 
One starts with 0 and grows the field every time a new frame shall be stored. 
Unlike the ``shape`` argument the ``chunk`` argument must not have 0
elements. In the example above we make the chunk size exactly one image frame
which is a reasonable choice for most applications. 
Finally the above example adds a deflate filter to the field (the same filter
used for gzip) with a compression level of 5. 

To create fields including non-existing intermediate groups use the
``create_field`` function. We could rewrite the last example like this

.. code-block:: python
    
    import pni.io.nx.h5 as nx

    file = nx.open_file("data.nxs",readonly=False)
    root = file.root();

    path = "/entry:NXentry/instrument:NXinstrument/detector:NXdetector"

    #create a deflate filter with compression level 5
    filter = nx.deflate_filter(5)

    field = nx.create_field(root, path, type="uint16", shape=(0,1024,2048),  
                            chunk=(1,1024,2048), filter = filter)

In this case none of the intermediate groups (``entyr``, ``instrument``,
 or ``detector``) must exist. They will be created automatically.


Field inquiry and manipulation
------------------------------

Like groups, fields have some public attributes which can be used to determine
some of the properties of a field 

==============  =========================================================
Attribute name  Description 
==============  =========================================================
``name``        returns the name of the group 
``path``        returns the full nexus path of the group
``parent``      returns the groups parent group 
``size``        number of elements stored in the field 
``type``        data type of the elements 
``shape``       a tuple with the number of elements along each dimension 
==============  =========================================================

Unlike numpy arrays, the shape of fields cannot be manipulated arbitrarily. 
This is due to limitations of the underlying HDF5 file format. 
Fortunately this is not a big issue. The most important manipulation (and
currently also the only one implemented) is to ``grow`` a field. 

.. code-block:: python
    
    #create the data field for the detector
    field = detector.create_field("data",...)

    #main measurement loop
    while True:
        data = get_data(...)    #retrieve data

        field.grow(0,1)         #grow field along dimension 0 by 1 element
       
        write_data(...)         #save data in newly appended slot
       
        #break the loop if the measurement is done
        if not measurement_running: break


The grow method of a field takes two positional arguments: the first is the 
dimension along we want to grow the field and the second is the number of 
elements by which we want to enlarge it.

**TODO: need a figure here showing how growing works**

Reading and writing data
------------------------

Fields behave a little like numpy arrays with the exception that the data is not
in memory but stored on disk. Reading and writing data works like with numpy
arrays. The best way to understand how this works is to have a look on a small
example. 
The next code snipped shows a typical use case where a bunch of image frames is
stored in a field and we want to iterate over the individual frames. 
The code should be rather self explaining

.. code-block:: python
    
    import pni.io.nx.h5 as nx

    file = nx.open_file("run_01.nxs")
    root_group = file.root()

    frame_path = "/:NXentry/:NXinstrument/:NXdetector/data"

    #retrieve frames from the file
    frames = nx.get_object(root_group,frame_path)

    #iterate over the frames
    for frame_index in frames.shape[0]:
        frame_data = frames[frame_index,...]
        result = do_som_work(frame_data)

Note here that the ellipse ``...`` used for retrieving the data will make 
the code independent from the actual rank of a frame.

Reading works just the other way around by assigning a numpy array or a scalar
or a list to a field. In the next snipped the example from the section about
field manipulation will be completed 

.. code-block:: python

    #create the data field for the detector
    field = detector.create_field("data",...)

    #main measurement loop
    while True:
        data = get_data(...)    #retrieve data

        field.grow(0,1)         #grow field along dimension 0 by 1 element
       
        field[-1,...] = data    #save data in newly appended slot
       
        #break the loop if the measurement is done
        if not measurement_running: break