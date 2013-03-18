#!/usr/bin/env python

import sys
import numpy

#sys.path.append("lib/python")

import pni.io.nx.h5 as nx

f = nx.create_file("test3.h5",True,0);

g = f.create_group("scan_1/detector/data")
sa= g.create_field("ListofStrings","string",[3,2])


sa[0,0]="safdfdsffdsfd"
sa[1,0]="safdsfsfdsffdsfd"
sa[2,0]="safdfsfd"

print sa[0,0]
print sa[1,0]
print sa[2,0]



#ww=sa.read()
#print ww

f.close()


