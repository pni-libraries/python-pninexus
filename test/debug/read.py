from __future__ import print_function
import pni.io.nx.h5 as nexus
import numpy

f = nexus.create_file("read.nxs", True)
r = f.root()
d = r.create_field("test_1", "string")
d.write("hello")
print(d.read())

d = r.create_field("test_2", "uint8")
d.write(1)
print(d.read())

d = r.create_field("test_3", "uint16", shape=(3,))
d.write(numpy.array([1, 2, 3]))
print(d.read())

d = r.create_field("test_4", "string", shape=(3,))
d.write(numpy.array(["hell", "wo", "t"]))
print(d.read())
