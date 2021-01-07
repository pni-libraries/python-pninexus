=================================
:py:mod:`pninexus.h5cpp.property`
=================================

.. automodule:: pninexus.h5cpp.property

Enumerations
============

.. autoclass:: DatasetFillValueStatus

   Determines the status of the fill value for a given dataset. 
   
   .. autoattribute:: DatasetFillValueStatus.UNDEFINED
      :annotation: no fill value is defined
      
   .. autoattribute:: DatasetFillValueStatus.DEFAULT
      :annotation: a default fill value has been set
      
   .. autoattribute:: DatasetFillValueStatus.USER_DEFINED
      :annotation: a fill value has been provided by the user
      
.. autoclass:: DatasetFillTime

   Determines whenthe fill value for a dataset will be set 
   
   .. autoattribute:: DatasetFillTime.IFSET
      :annotation: fill value will be set during setup
      
   .. autoattribute:: DatasetFillTime.ALLOC
      :annotation: fill value will be set during allocation
      
   .. autoattribute:: DatasetFillTime.NEVER
      :annotation: the fill value will never be set
      
.. autoclass:: DatasetAllocTime

   Enumeration determining the point in time when space for a dataset will be 
   allocated. 
   
   .. autoattribute:: DatasetAllocTime.DEFAULT
      :annotation: use the default allocation time
   
   .. autoattribute:: DatasetAllocTime.EARLY
      :annotation: as early as possible 
   
   .. autoattribute:: DatasetAllocTime.INCR
      :annotation: incremental
   
   .. autoattribute:: DatasetAllocTime.LATE
      :annotation: as late as possible
   
   
.. autoclass:: DatasetLayout

   Enumeration determining the layout of an HDF5 dataset 
   
   .. autoattribute:: DatasetLayout.COMPACT
      :annotation: use a compact layout
   
   .. autoattribute:: DatasetLayout.CONTIGUOUS
      :annotation: use a contiguous layout
   
   .. autoattribute:: DatasetLayout.CHUNKED
      :annotation: use a chunked layout

   .. autoattribute:: DatasetLayout.VIRTUAL
      :annotation: use a virtual layout
   
   
.. autoclass:: LibVersion

   .. autoattribute:: LibVersion.LATEST
      :annotation: denotes the latest library version
   
   .. autoattribute:: LibVersion.EARLIEST
      :annotation: denotes the earliest library version
   
.. autoclass:: CopyFlag

   Flag controlling the copying behavior in HDF5.

   .. autoattribute:: SHALLOW_HIERARCHY
      :annotation: no recursive copying
   
   .. autoattribute:: EXPAND_SOFT_LINKS
      :annotation: expand soft links when copying
   
   .. autoattribute:: EXPAND_EXTERNAL_LINKS
      :annotation: expand external links when copyin
   
   .. autoattribute:: EXPAND_REFERENCES
      :annotation: expand references when copying
   
   .. autoattribute:: WITHOUT_ATTRIBUTES
      :annotation: do not include attributes when copying
   
   .. autoattribute:: MERGE_COMMITTED_TYPES
      :annotation: merge committed types during copying 
   
Utility classes
===============
   
 
.. autoclass:: CopyFlags
   :members:
   :undoc-members:

.. autoclass:: ChunkCacheParameters
   :members:
   :undoc-members:
 
.. autoclass:: CreationOrder
   :members:
   :undoc-members:
 
Property list classes
=====================
 
.. autoclass:: List
   :members:
   :undoc-members:
 
   Base class for all property lists. 
   
.. autoclass:: StringCreationList
   :members:
   :undoc-members:
   
.. autoclass:: AttributeCreationList
   :members:
   :undoc-members:
   
Object property lists
---------------------

.. autoclass:: ObjectCopyList
   :members:
   :undoc-members:
   
.. autoclass:: ObjectCreationList
   :members:
   :undoc-members:
   
Dataset property lists
----------------------
   
.. autoclass:: DatasetCreationList
   :members:
   :undoc-members:
   
.. autoclass:: DatasetTransferList
   :members:
   :undoc-members:

   Property list controlling the data transfer during dataset IO operations.
   
.. autoclass:: DatasetAccessList
   :members: 
   :undoc-members:
   
   Property list controlling access to a dataset
   
File property lists
-------------------
   
.. autoclass:: FileAccessList
   
   Property list controlling how files are accessed. 
   
   .. autoattribute:: library_version_bound_high
   
      Read-only property returning the version uppermost version boundary. 
      
      :return: library boundary
      :rtype: LibVersion
   
   .. autoattribute:: library_version_bound_low
   
      Read-only property returning the lowest compatible HDF5 version. 
      
      :return: library bound
      :rtype: LibVersion
   
   .. automethod:: library_version_bounds(lower,upper)
   
      Set the upper and lower compatibility version bounds for a new file. 
      
      :param LibVersion lower: determines the lowest compat. version
      :param LibVersion upper: determines the highest compat. version
      
   
.. autoclass:: FileCreationList
   :members:
   :undoc-members:
   
   Property list controlling the file creation process
   

   
.. autoclass:: FileMountList
   :members:
   :undoc-members:
   
   Property List controlling the mounting of external files. 
   
   
Group property lists
--------------------

.. autoclass:: GroupCreationList
   :members:
   :undoc-members:
   
.. autoclass:: GroupAccessList
   :members:
   :undoc-members:
   
Link property lists
-------------------

.. autoclass:: LinkCreationList
   :members:
   :undoc-members:
   
.. autoclass:: LinkAccessList
   :members:
   :undoc-members:
   
Type property lists
-------------------

.. autoclass:: TypeCreationList
   :members:
   :undoc-members:
   
.. autoclass:: DatatypeAccessList
   :members:
   :undoc-members:

 
 
