#!/usr/bin/env python
#File: nxattr_ex2.py

import numpy
import pni.io.nx.h5 as nx

nxfile = nx.create_file("nxgroup_ex3.h5",overwrite=True)
g = nxfile.create_group("scan_1","NXentry")
g.attr("description","string").value = "a first measurement"
g.attr("temperature","float64").value = 123.3
g.attr("velocity","float64",shape=(3,)).value = numpy.arange(0,3,dtype="float64")

#iterate over all attributes of the file object
print "\nFile attribute ......"
for a in nxfile.attributes:
    print a.name,": ",a.value

print "\n\ngroup attribute ......"
for a in g.attributes:
    print a.name,": ",a.value



