#!/usr/bin/env python 

import sys

sys.path.append("lib/python")

import pni.nx.h5.nxh5 as nx

f = nx.create_file("test.h5",True,0);
g = f.create_group("scan_1")
