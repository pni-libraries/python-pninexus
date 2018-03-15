from __future__ import print_function
from pninexus import  h5cpp
from pninexus.h5cpp.node import Dataset
from pninexus.h5cpp.file import AccessFlags
from pninexus.h5cpp.dataspace import Simple,UNLIMITED,Hyperslab
from pninexus.h5cpp.property import LinkCreationList,DatasetCreationList,DatasetLayout
import numpy

#
# turn off HDF5 error output
#
h5cpp.print_hdf5_errors(False)

h5file = h5cpp.file.create("dataset.h5",AccessFlags.TRUNCATE)
root   = h5file.root()

#
# create an extensible dataset (we use single values here for the sake of 
# simplicity)
#
dataspace = Simple((0,),(UNLIMITED,))
datatype = h5cpp.datatype.kFloat32
lcpl = LinkCreationList()
dcpl = DatasetCreationList()
dcpl.layout=DatasetLayout.CHUNKED
dcpl.chunk = (1024*1024,)
dataset  = Dataset(root,h5cpp.Path("data"),datatype,dataspace,lcpl,dcpl)

#
# writing data sequentialy to the dataset
#
selection = Hyperslab(offset=(0,),block=(1,))
for index in range(0,20):
    selection.offset(0,index)
    dataset.extent(0,1)
    dataset.write(index*0.03,selection)
    
#
# read everything
#
data = dataset.read()
print(data)

#
# read individual numbers back
#
selection = Hyperslab((0,),(1,))
while(True):
    try:
        print(dataset.read(selection=selection))
    except:
        break
    
    selection.offset(0,selection.offset()[0]+1)
    
    