#!/usr/bin/env python 

import sys
import numpy
import pni.io.nx.h5 as nx

f = nx.create_file("test.h5",True,0);
d = f.attr("runnumber","int32")
d.value = 1209
print d.value," shoule be 1209"

g = f.create_group("scan_1/detector/data")

d = g.attr("description","string")
d.value = "hello world"
print d.value

g.attr("temperature","float32",[4,2,3])
g.link("/scan_1/detector/data","/data")

a = g.attr("temperature")
t = numpy.zeros(a.shape,a.dtype)
t[0,:,:] = 17.2
t[1,:,:] = 18.2
t[2,:,:] = 19.2
t[3,:,:] = 20.2
a.value = t 
print a.value
print a.value[0,:,:]

a = f.attr("runnumber")
print a.value

g = f.open("/scan_1/detector/data")
g.attr("ref_index","complex64").value = 1.0548e-4+2.123e-6j
print g.attr("ref_index").value


f.close()


