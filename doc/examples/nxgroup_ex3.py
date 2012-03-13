#!/usr/bin/env python
#File: nxgroup_ex2.py

import numpy
import pni.nx.h5 as nx

nxfile = nx.create_file("nxgroup_ex3.h5",overwrite=True)
g = nxfile.create_group("scan_1","NXentry")
g.create_group("instrument","NXinstrument")
g.create_group("sample","NXsample")
g.create_group("data","NXdata")
g.create_field("experiment_description","string").write("SI0815")
print g.nchilds

for i in range(g.nchilds):
    print g[i],g[i].path


nxfile.close()
