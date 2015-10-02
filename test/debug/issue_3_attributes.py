import pni.io.nx.h5 as nexus
import numpy

f = nexus.create_file("issue_3_attribute.nxs",True)
r = f.root()

shape = (1,10)
a = r.attributes.create("test","uint16",shape=shape)
d = numpy.ones(shape,dtype="uint16")
a[...] = d
