#!/usr/bin/env python
#File: attribute_access.py
from __future__ import print_function
import pni.io.nx.h5 as nexus 

f = nexus.create_file("attributes_access.nxs",overwrite=True)
r = f.root()

#access attributes via their index
print("Access attributes via index")
for index in range(len(r.attributes)):
    print("{index}: {name}".format(index=index,
                                   name=r.attributes[index].name))

#access attributes via iterator
print()
print("Access attributes via iterator")
for attr in r.attributes:
    print(attr.name)

#access directly via name 
print(r.attributes["NeXus_version"].name)

