#!/usr/bin/env python
import pni.nx.h5 as nx

nxfile = nx.create_file("nxgroup_ex1.h5",overwrite=True)

#create groups
nxfile.create_group("scan_1","NXentry")
nxfile.create_group("scan_1/instrument","NXinstrument")
nxfile.create_group("scan_1/sample","NXsample")
nxfile.create_group("scan_1/monitor","NXmonitor")

#iterate over groups
scan = nxfile["scan_1"]
for g in scan.childs:
    print g.path


nxfile.close()
