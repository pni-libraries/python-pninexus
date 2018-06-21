#!/usr/bin/env python

import nx as nexus
import os
import sys

pid = os.getpid()
mfile = "/proc/%i/statm" % (pid)
lfile = "mem_check_{pid}.log".format(pid=pid)
ofile = "mem_check_{pid}.nxs".format(pid=pid)


def test_function(nxfile):
    r = nxfile.root()
    g = r.create_group("detector")
    f = r.create_field("name", "string")
    attr = f.attributes.create("temperature", "float32")
    attr.write(1.23)
    attr.close()
    g.close()
    f.close()
    r.close()


nruns = int(sys.argv[1])


with open(lfile, "w") as mlog:

    for i in range(nruns):
        logline = "%i " % (i)

        with open(mfile) as stat_file:
            logline += stat_file.readline()

        mlog.write(logline)
        # call the test function
        nxfile = nexus.create_file("test_mem.nxs", True)
        test_function(nxfile)
        nxfile.close()


sys.stdout.write(lfile + "\n")
