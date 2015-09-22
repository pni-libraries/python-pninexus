#!/usr/bin/env python
#File: nxattr_ex1.py

from __future__ import print_function
import pni.io.nx.h5 as nexus

f = nexus.create_file("nxattr_ex1.h5",overwrite=True)
r = f.root()

fmt = "{name} : {path} type={dtype}"

for a in r.attributes:
    print(fmt.format(name=a.name,path=a.path,dtype=a.dtype))

f.close()
