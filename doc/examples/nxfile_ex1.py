#!/usr/bin/env python
#file: nxfile_ex1.py

import pni.io.nx.h5 as nx

nxfile = nx.create_file("nxfile_ex1.h5",overwrite=True)
nxfile.close()

