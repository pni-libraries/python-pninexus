#!/usr/bin/env python
import numpy
import pni.nx.h5 as nx

def write_data(fname,np,nxpt,nypt):
    frame = numpy.zeros((nxpt,nypt),dtype="uint16")

    nxfile = nx.create_file(fname,overwrite=True)

    g = nxfile.create_group("/scan_1/instrument/detector","NXdetector");
    data = g.create_field("data","uint16",[0,nxpt,nypt])
    print data.shape

    for i in range(np):
        data.grow(0)
        frame[...] = i
        data[i,...] = frame

    #close everything
    g.close()
    data.close()
    nxfile.close()

def read_data(fname):

    nxfile = nx.open_file(fname,readonly=False);
    field = nxfile["/scan_1/instrument/detector/data"];

    data = numpy.zeros(field.shape[1:],dtype=field.type_id)
   
    for i in range(field.shape[0]):
        data = field[i,...]


    #close everything
    field.close()
    nxfile.close()

fname="simple_io.h5"
nxpt = 10
nypt = 20
np = 100

print "writing data ..."
write_data(fname,np,nxpt,nypt)
print "reading data ..."
read_data(fname)
print "program finished ..."
