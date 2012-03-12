
import numpy
import pni.nx.h5 as nx

def write_data(fname,np,nx,ny):
    frame = numpy.zeros((nx,ny),dtype="uint16")

    nxfile = nx.create_file(fname,True,0)

    g = nxfile.create_group("/scan_1/instrument/detector","NXdetector");
    data = g.create_field("data","uint16",(0,nx,ny))

    for i in range(np):
        data.grow(0)
        frame[...] = i
        data[i,...] = frame

    #close everything
    g.close()
    data.close()
    nxfile.close()

def read_data(fname):

    nxfile = nx.open_file(fname);
    field = nxfile["/scan_1/instrument/detector/data"];

    data = numpy.zeros((field.shape[1],field.shapep[2]),dtype=field.type_id)
   
    for i in range(field.shape[0]):
        data = field[i,...]


    #close everything
    field.close()
    nxfile.close()

fname="simple_io.h5"
nx = 10
ny = 20
np = 100

print "writing data ..."
wite_data(fname,np,nx,ny)
print "reading data ..."
read_data(fname)
print "program finished ..."
