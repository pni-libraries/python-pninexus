from __future__ import print_function
import pni.io.nx.h5 as nexus
import numpy

f = nexus.create_file("attributes_all.nxs", True)
r = f.root()

print(r.attributes["HDF5_Version"].read())

a = r.attributes.create("temp", "uint32", shape=(2, 3))
a.write(numpy.array([[1, 2, 3], [4, 5, 6]]))
print(a.read())
