#!/usr/bin/env python
from __future__ import print_function
import numpy
import pni.io.nx.h5 as nexus

file_struct = \
"""
<group name="entry" type="NXentry">
    <group name="instrument" type="NXinstrument">
        <group name="detector" type="NXdetector">
        </group>
    </group>
</group>
"""

class detector_description(object):
    def __init__(self,nx,ny,dtype):
        self.shape = (nx,ny)
        self.dtype = dtype

def init_file(fname):
    f = nexus.create_file(fname,overwrite=True)
    r = f.root()
    nexus.xml_to_nexus(file_struct,r)

    return f

def write_data(f,detector,np):

    frame_buffer = numpy.zeros(detector.shape,dtype=detector.dtype)
    r = f.root()
    detector_group = nexus.get_object(r,"/:NXentry/:NXinstrument/:NXdetector")
   
    data = detector_group.create_field("data",detector.dtype,
                                       shape=(0,detector.shape[0],detector.shape[1]))
    print(data.shape)
    for i in range(np):
        data.grow(0)
        frame_buffer[...] = i
        data[-1,...] = frame_buffer


def read_data(f):
    
    data = nexus.get_object(f.root(),"/:NXentry/:NXinstrument/:NXdetector/data")
   
    for i in range(data.shape[0]):
        print(data[i,...])


fname="simple_io.nxs"
det  = detector_description(10,20,"uint16")

nxfile = init_file(fname)

print("writing data ...")
write_data(nxfile,det,100)
print("reading data ...")
read_data(nxfile)
print("program finished ...")
