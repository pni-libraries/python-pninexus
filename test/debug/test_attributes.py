from __future__ import print_function

import pni.io.nx.h5 as nx
import numpy

f = nx.create_file("test_attributes.nxs", True)
r = f.root()
a = r.attributes.create("test_scalar", "string")
a[...] = "hello world"
a = r.attributes.create("test_array", "string", shape=(2, 3))
data = numpy.array([["hello", "world", "this"], ["is", "a", "test"]])
print(data)
print(data.dtype)
a[...] = data
a.close()
r.close()
f.close()

f = nx.open_file("test_attributes.nxs")
r = f.root()
a = r.attributes["test_scalar"]
print(a[...])

a = r.attributes["test_array"]
print(a[...])
print(a[...].flat[...])
