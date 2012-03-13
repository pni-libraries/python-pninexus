#!/usr/bin/env python
#File: nxfile_ex1.cpp

import pni.nx.h5 as nx

nxfile = nx.create_file("nxfile_ex1.h5",overwrite=True)
nxfile.close()

