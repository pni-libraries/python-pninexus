#!/usr/bin/env python 

import sys

sys.path.append("lib/python")

import pni.nx.h5.nxh5 as nx

f = nx.create_file("test.h5",True,0);
d = f.attr("runnumber","int32")
d.write(1209)
g = f.create_group("scan_1/detector/data")
d = g.attr("description","string")
print d.shape
print d.type_id
d.write("hello world")
g.link("/scan_1/detector/data","/data")

g = f.open("scan_1")
g.create_group("motors")

f.close()


