from __future__ import print_function
from pni.io import h5cpp
from pni.io import nexus
import time
import sys

def create_file_property_lists():
   
    fcpl = h5cpp.property.FileCreationList()
    fapl = h5cpp.property.FileAccessList()
    fapl.library_version_bounds(h5cpp.property.LibVersion.LATEST,
                                h5cpp.property.LibVersion.LATEST)
    
    return (fcpl,fapl) 

def open_swmr_file(filename):
    
    fcpl,fapl = create_file_property_lists()
    
    flags = h5cpp.file.AccessFlags.READONLY | h5cpp.file.AccessFlags.SWMRREAD
    print(h5cpp.file.AccessFlags.READONLY)
    print(h5cpp.file.AccessFlags.SWMRREAD)
    print(flags)
    f = h5cpp.file.open(filename,h5cpp.file.AccessFlags.READONLY | h5cpp.file.AccessFlags.SWMRREAD,fapl)
    
    return f


nxfile = open_swmr_file("test.nxs")

detectors = nexus.get_objects(nxfile.root(),nexus.Path.from_string("/:NXentry/:NXinstrument/:NXdetector/data"))
if len(detectors)!=1:
    print("Too much or to less detectors found")
    sys.exit(1)
    
detector_data = detectors[0]

current_offset = 0
selection = h5cpp.dataspace.Hyperslab(offset=(current_offset,),block=(1,))

for index in range(100):
    detector_data.refresh()
    current_block = detector_data.dataspace.size-current_offset
    selection.offset(0,current_offset)
    selection.block(0,current_block)
    print(current_offset)
    #selection.offset(0,current_index)
    print(detector_data.read(selection=selection))
    
    current_offset = detector_data.dataspace.size
    
    time.sleep(3)