#!/usr/bin/env python 

import sys

sys.path.append("lib/python")

import pni.nx.h5 as nx

f = nx.create_file("test.h5",True,0);
d = f.attr("runnumber","int32")
d.write(1209)

g = f.create_group("scan_1/detector/data")

d = g.attr("description","string")

print d.shape
print d.type_id
d.write("hello world")

g.attr("temperature","float32",[1,2,3])
g.link("/scan_1/detector/data","/data")

a = g.attr("temperature")
print a.shape
print a.type_id

a = f.attr("runnumber")
print a.shape
print a.type_id

g = f.open("/scan_1/detector/data")
g.attr("ref_index","complex64").write(1e-4+1e-6j)


f.close()


