from __future__ import print_function

import pni.io.nx.h5 as nx
import numpy

f = nx.create_file("test_fields.nxs", True)
r = f.root()
field = r.create_field("test_scalar", "string")
field[...] = "hello world"
field = r.create_field("test_array", "string", shape=(2, 3))
data = numpy.array([["hello", "world", "this"], ["is", "a", "test"]])
print(data)
print(data.dtype)
field[...] = data
field.close()
r.close()
f.close()

f = nx.open_file("test_fields.nxs", False)
r = f.root()
field = r["test_scalar"]
print(field[...])

field = r["test_array"]
print(field[...])
