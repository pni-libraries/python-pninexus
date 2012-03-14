#!/usr/bin/env python
#File: nxgroup_ex3.py

import numpy
import pni.nx.h5 as nx

nxfile = nx.create_file("nxgroup_ex3.h5",overwrite=True)
g = nxfile.create_group("scan_1","NXentry")
g.create_group("instrument","NXinstrument")
g.create_group("sample","NXsample")
g.create_group("data","NXdata")
g.create_field("experiment_description","string").write("SI0815")
print g.nchilds

print "Iterate over file childs ..."
for c in nxfile.childs:
    print c.path

print "Iterate over group childs ..."
for i in range(g.nchilds):
    print g[i],g[i].path

for c in g.childs:
    print c,c.path

nxfile.close()
