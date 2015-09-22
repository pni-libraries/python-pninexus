#!/usr/bin/env python
#File: nxattr_ex2.py
from __future__ import print_function
import numpy
import pni.io.nx.h5 as nexus 

f = nexus.create_file("nxattr_ex2.nxs",overwrite=True)
g = f.root().create_group("scan_1","NXentry")

g.attributes.create("description","string")[...] = "a first measurement"
g.attributes.create("temperature","float64")[...] = 123.3
g.attributes.create("velocity","float64",shape=(3,))[...] = numpy.arange(0,3,dtype="float64")

output = "{name} = {data}"
#print the results
for a in g.attributes:
    print(output.format(name=a.name,data=a[...]))




