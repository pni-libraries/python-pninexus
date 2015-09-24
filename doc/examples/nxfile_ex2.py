#!/usr/bin/env python
#file: nxfile_ex2.py

import pni.io.nx.h5 as nexus

fname = "nxfile_ex2.nxs"
try:
    print "try to open an existing file ..."
    nxfile = nexus.open_file(fname)
except:
    print "Error opening file - recreate ..."
    nxfile = nexus.create_file(fname,overwrite=True)


nxfile.close()

