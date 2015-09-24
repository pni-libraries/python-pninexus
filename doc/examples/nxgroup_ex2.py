#!/usr/bin/env python
#file: nxgroup_ex2.py
from __future__ import print_function
import pni.io.nx.h5 as nexus

nxfile = nexus.create_file("nxgroup_ex2.nxs",overwrite=True)

#create groups
entry = nxfile.root().create_group("scan_1","NXentry")
entry.create_group("instrument",'NXinstrument')
entry.create_group("sample","NXsample").\
      create_group("transformations","NXtransformation")
entry.create_group("control","NXmonitor")

#iterate over groups
for child in entry.recursive:
    print(child.path)


nxfile.close()
