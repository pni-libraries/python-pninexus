===============================
:py:mod:`pni.io.h5cpp.property`
===============================

.. automodule:: pni.io.h5cpp.property

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
   
   
.. autoclass:: LibVersion

   .. autoattribute:: LibVersion.LATEST
   
   .. autoattribute:: LibVersion.EARLIEST