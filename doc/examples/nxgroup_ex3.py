#!/usr/bin/env python
#File: nxgroup_ex3.py

import numpy
import pni.nx.h5 as nx

nxfile = nx.create_file("nxgroup_ex3.h5",overwrite=True)
g = nxfile.create_group("scan_1","NXentry")
g.create_group("instrument","NXinstrument")
g.create_group("sample","NXsample")
g.create_group("data","NXdata")
f = g.create_field("experiment_description","string")
f.write("SI0815")


#iterate over all attributes of the file object
for a in nxfile.attributes:
    print a.name,": ",a.value


print "Iterate over file childs ..."
for c in nxfile.childs:
    print c.path

print "Iterate over group childs ..."
for c in g.childs:
    print c,c.path
    for a in c.attributes:
        print a.name,": ",a.value

