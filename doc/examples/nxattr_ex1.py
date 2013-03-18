#!/usr/bin/env python
#File: nxattr_ex1.py

import numpy
import pni.io.nx.h5 as nx

nxfile = nx.create_file("nxattr_ex1.h5",overwrite=True)
g = nxfile.create_group("data");

#string attributes
attr = g.attr("description","string")
attr.value = "a stupid data gruop"
print attr.value

#a float attribute
attr = g.attr("temperature","float32")
attr.value = 389.2343
print attr.value

#a array attribute
wa = numpy.zeros((10,3),dtype="uint8")
wa[...] = 10
g.attr("vectors","uint8",wa.shape).value = wa
print g.attr("vectors").value

nxfile.close()
