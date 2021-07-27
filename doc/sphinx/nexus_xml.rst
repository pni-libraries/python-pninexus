Nexus and XML 
=============

Though NeXus files can be rather complicated, accessing data can be quite easy
as shown in the previous sections of this manual. 
However, the complexity of NeXus files can become rather painful when creating
file. This is an issue which typically affects developers working on data
acquisition software. 
The ``python-pninexus`` framework provides a means of generating skeleton NeXus
files from XML. Even existing files can be extended by structures described by
XML. 

A NeXus XML primer 
------------------

The XML dialect used to create or extend files with ``python-pninexus`` is very
close to NXDL, the XML language used to define classes in the NeXus standard. 
However, NXDL is not powerful enough for creating HDF5 files as it lacks some
tags (for instance tags for describing the chunk-shape of a field) and 
the types are not specific enough (`NX_FLOAT` would be valid for 32 and 64 Bit
floating point numbers). 

Creating groups
~~~~~~~~~~~~~~~

The `group` tag is used to create groups which can then embed other groups,
fields, or attributes. Its usage is fairly simple

.. code-block:: xml

    <group name="scan_1" type="NXentry">
    </group>

The `name` attribute of the tag describes the name of the group to create 
and the `type` attribute the NeXus base class (this value will be stored in the 
`NX_class` attribute of the HDF5 group created).

A typical application for groups would be the construction of a basic NeXus
skeleton 

.. code-block:: xml

    <group name="scan_1" type="NXentry">
        <group name="instrument" type="NXinstrument">
            <group name="source" type="NXsource"/>
            <group name="detector" type="NXdetector"/>
        </group>

        <group name="sample" type="NXsample"/>
        <group name="data" type="NXdata"/>
        <group name="control" type="NXmonitor"/>
    </group>


Creating fields
~~~~~~~~~~~~~~~

Fields are described by the `field` tag. For a simple scalar field use 

.. code-block:: xml

    <field name="data" type="float64"/>

Like for the `group` tag the `name` attribute describes the name of the field
to create and the `type` tag its data type. For the data type the standard
:py:mod:`numpy` type strings are used.

For a multidimensional field we have to embed the `dimensions` tag in the field 

.. code-block:: xml

    <field name="data" type="uint32">
        <dimensions rank="3">
            <dim index="1" value="0"/>
            <dim index="2" value="1024"/>
            <dim index="3" value="512"/>
        </dimensions>
    </field>

The attribute `rank` of the `dimensions` tag stores the number of dimensions
the resulting field should have. The attributes of the `dim` tag for every
dimension should be rather self-explaining. There are however two things to
note here: the dimension index starts with 1 (unlike in C with 0) and
dimensions with 0 elements are allowed (one can later grow the field as we have
already seen). By default the chunk shape matches the field shape with the
first element set to 1. In the above example the chunk shape would be
(1,1024,512). 

In order to explicitly set the chunk shape use the `chunk` tag 

.. code-block:: xml
    
    <field name="data" type="uint32">
        <dimensions rank="3">
            <dim index="1" value="0"/>
            <dim index="2" value="1024"/>
            <dim index="3" value="512"/>
        </dimensions>
        <chunk rank="3">
            <dim index="1" value="1"/>
            <dim index="2" value="512"/>
            <dim index="3" value="512"/>
        </chunk>
    </field>

The `chunk` tag is currently not implemented due to limitations of the
underlying `libpninexus`. 

Finally the `field` tag accepts an additional attribute `units` which stores
the physical unit of the data stored in a field. 

.. code-block:: xml

    <field name="distance" type="float32" units="m"/>

The value will be stored in a string attribute `units` attached to the created
HDF5 field. 


Dealing with attributes
~~~~~~~~~~~~~~~~~~~~~~~

Attributes are quite similar to fields but created with the `attribute` tag. 
An attribute tag can appear within a `field` or `group` tag. Unlike the `field`
tag the `attribute` tag does not have a `units` attribute. 
The chunk shape cannot be set and no compression for attribute data is
available. 
However, multidimensional attributes can be created just like fields by
embedding an `dimensions` tag within the `attribute` tag.  
Here is a short example with an attribute attached to a field

.. code-block:: xml

    <field name="chi" type="float64" units="mm">
        <attribute name="transformation_type" type="string"/>
        <attribute name="vector" type="float32">
            <dimensions rank="1"> 
                <dim index="1" value="3"/>
            </dimensions>
        </attribute>
    </field>

Writing data from XML
~~~~~~~~~~~~~~~~~~~~~

The last example already shows a problem: it many cases it would be feasible to 
not only create a field or attribute but also to fill it with data. This is
particularly true when the field or attribute should store static data which
does not change during an experiment. 

:py:mod:`python-pninexus` supports this feature. Full support is provided for 
numeric types. In the above example the `vector` attribute could be filled with
data with

.. code-block:: xml

    <attribute name="vector" type="float32">
        <dimensions rank="1">
            <value index="1" value="3"/> 
        </dimensions>
        0.0 0.0 1.0
    </attribute>

to denote a rotation around the z-axis. For string types only scalar data is
currently supported. For the previously defined `transformation_type` attribute 
we could set the data with 

.. code-block:: xml

    <attribute name="transformation_type" type="string">rotation</attribute> 

From the above examples it is clear that :py:mod:`python-pninexus` does not follow
the standard NXDL convention for denoting data within a field or attribute tag. 
The reason for this is to make the resulting XML more readable. 
However, this comes at a price that strings are handled different from numeric
values. The reason for this possibly unexpected behavior is that numeric values
can easily be parsed and thus can be written as a block of whitespace delimited
values. For obvious reasons this is not true for strings as
whitespace-characters can be part of the string. However, for most applications 
this limitation is not a serious problem.

Creating links
~~~~~~~~~~~~~~

Links can be created using the `link` tag. External as well as internal links
are supported. An external link can be made like this 

.. code-block:: xml

    <group name="entry" type="NXentry">
        <group name="instrument" type="NXinstrument">
            <group name="detector" type="NXdetector">
                <field name="data" type="uint32" units="cps">
                    <dimensions rank="2">
                        <dim index="1" value="0"/>
                        <dim index="2" value="1024"/>
                    </dimensions>
                </field>
            </group>
        </group>

        <group name="data" type="NXdata">
            <link name="data" target="/entry/instrument/detector/data"/>
        </group>
    </group>

The link below the `NXdata` group refers to the `data` field in the detector
class. 

The only thing we have to change for an external link is the target path. 
We can modify the above example for the case where the detector data is stored
in a different file like this 

.. code-block:: xml

    <group name="entry" type="NXentry">
        <group name="data" type="NXdata">
            <link name="data" target="detector_file.nxs://entry/instrument/detector/data"/>
        </group>
    </group>

From XML to NeXus
-----------------

To create a NeXus file from XML is rather simple. Just use the
:py:func:`xml_to_nexus` function provided by the package 

.. code-block:: python

    import pninexus.nexus as nexus

    xml_struct = \
    """
    <group name="entry" type="NXentry">
            ......
    </group>
    """

    f = nexus.create_file("test.nxs",overwrite=True)
    r = f.root()

    nexus.xml_to_nexus(xml_struct,r)

The first argument to :py:func:`xml_to_nexus` is the XML string from which acts
as a blue-print for the structure to create. The second argument is the parent
object below which the structure should be created. 

In the above example no data would be written to the file. This is due to a
missing third argument to :py:func:`xml_to_nexus`: the write predicate. 
One may does not want to write all the data from the XML to the file. It is
therefore possible to pass a predicate function which decides whether or not
the data for a particular offset should be written to disk. 

If all data should be written we can use something like the following 

.. code-block:: python
    
    nexus.xml_to_nexus(xml_struct,r,lambda obj: True)

The predicate function takes a single argument, the currently created NeXus
object, and decides then whether or not data should be written by returning
:py:const:`True` or :py:const:`False`.
If only fields of size one should be written we can use the following approach 

.. code-block:: python

    def write_pred(obj):
        return isinstance(obj,nexus.nxfield) and obj.size == 1

    nexus.xml_to_nexus(xml_struct,r,write_pred)


