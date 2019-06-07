#!/usr/bin/env python
#File: attribute_io.py
from __future__ import print_function
import numpy
import pni.io.nx.h5 as nexus 

f = nexus.create_file("attribute_io.nxs",overwrite=True)
samptr = f.root().create_group("scan_1","NXentry"). \
                  create_group("instrument","NXinstrument"). \
                  create_group("sample","NXsample"). \
                  create_group("transformations","NXtransformations")

samptr.parent.create_field("depends_on","string")[...]="transformation/tt"

tt = samptr.create_field("tt","float64",shape=(10,))
tt[...] = numpy.array([1,2,3,4,5,6,7,8,9,10])
tt.attributes.create("transformation_type","str")[...] = "rotation"
a = tt.attributes.create("vector","float64",shape=(3,))
a[...] = numpy.array([-1,0,0])

print("r=",a[...])
print("x=",a[0])
print("y=",a[1])
print("z=",a[2])

