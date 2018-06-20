from __future__ import print_function
import pni.io.nx.h5 as nexus

f = nexus.create_file("test.nxs", True)
r = f.root()

d = r.create_field("data", "string")
d[...] = "hello world"
print(d[...])
