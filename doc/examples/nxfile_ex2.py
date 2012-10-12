#!/usr/bin/env python

import pni.nx.h5 as nx

fname = "nxfile_ex2.h5"
try:
    print "try to open an existing file ..."
    nxfile = nx.open_file(fname)
except nx.NXFileError:
    print "Error opening file - recreate ..."
    nxfile = nx.create_file(fname,overwrite=True)


nxfile.close()

