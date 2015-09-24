#!/usr/bin/env python
from __future__ import print_function
import sys
import numpy
import pni.io.nx.h5 as nexus

f = nexus.create_file("test_string2.nxs",True);

d = f.root().create_group("scan_1","NXentry").\
             create_group("detector","NXdetector")
sa= d.create_field("ListofStrings","string",shape=(3,2))


sa[0,0]="safdfdsffdsfd"
sa[1,0]="safdsfsfdsffdsfd"
sa[2,0]="safdfsfd"

print(sa[0,0])
print(sa[1,0])
print(sa[2,0])


print(sa[...])

f.close()


