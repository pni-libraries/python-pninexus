#File: nxgroup_ex2.py

import numpy
import pni.nx.h5 as nx

nxfile = nx.create_file("nxgroup_ex2.h5",true,0);
g = file.create_group("data");

#string attributes
attr = g.attr("description")
attr.value = "a stupid data gruop"
print attr.value

#a float attribute
attr = g.attr("temperature")
attr.value = 389.2343
print attr.value

#a array attribute

wa = numpy.zeros((10,3),dtype="float32")
for(size_t i=0;i<wa.size();i++) wa[i] = i;
g.attr("vectors").value = wa
print g.attr("vectors")

nxfile.close()
