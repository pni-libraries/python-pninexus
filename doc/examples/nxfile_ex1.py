#!/usr/bin/env python
#file: nxfile_ex1.py

import pni.io.nx.h5 as nexus

nxfile = nexus.create_file("nxfile_ex1.nxs",overwrite=True)
nxfile.close()

