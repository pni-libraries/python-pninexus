#!/usr/bin/env python
import pni.nx.h5 as nx

nxfile = nx.create_file("nxgroup_ex1.h5",overwrite=True)

#create groups
nxfile.create_group("scan_1","NXentry")
inst = nxfile.create_group("scan_1/instrument","NXinstrument")

print "group name: ",inst.name
print "group path: ",inst.path
print "group base: ",inst.base

nxfile.close()
