from __future__ import print_function
import pni.io.nx.h5 as nexus
import numpy

f = nexus.create_file("issue_3_fields.nxs",True)
r = f.root()

shape = (1,10)
a = r.create_field("test","uint16",shape=shape)
d = numpy.ones(shape,dtype="uint16")

a[...] = d

shape = (10,1)
a = r.create_field("test2","uint16",shape=shape)
a[...] = d
