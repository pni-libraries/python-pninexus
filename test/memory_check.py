#!/usr/bin/env python

import pni.nx.h5 as nx
import os
import sys

pid = os.getpid()
mfile = "/proc/%i/statm" %(pid)
lfile = "mem_check_%i.log" %pid

mlog = open(lfile,"w")

def test_function(nxfile):
    g = nxfile.create_group("detector")
    f = nxfile.create_field("name","string")
    attr = f.attr("temperature","float32")
    attr.value = 1.23

    attr.close()
    g.close()
    f.close()

nruns = int(sys.argv[1])

for i in range(nruns):
    logline = "%i " %(i)
    f = open(mfile)
    logline += f.readline()
    f.close()
    mlog.write(logline)
    #call the test function
    nxfile = nx.create_file("test_mem.h5",True)
    test_function(nxfile)
    nxfile.close()

mlog.close()

sys.stdout.write(lfile)



