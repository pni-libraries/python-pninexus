import numpy
import pni.io.nx.h5 as nexus

f = nexus.create_file("scalar.nxs", True)
r = f.root()
d = r.create_field("data", "uint16")

d1 = 14
d.write(d1)
o = d.read()
print(d1, o)

d2 = numpy.array([16])
d.write(d2)
o = d.read()
print(d2, o)
