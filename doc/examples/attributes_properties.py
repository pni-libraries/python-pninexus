#!/usr/bin/env python
#File: attributes_properties.py

from __future__ import print_function
import pni.io.nx.h5 as nexus

f = nexus.create_file("attributes_properties.nxs",overwrite=True)
r = f.root()

fmt = "{name:20} : {path:20} type={dtype:<10} size={size:<20}"

for a in r.attributes:
    print(fmt.format(name=a.name,
                     path=a.path,
                     dtype=a.dtype,
                     size=a.size))

f.close()
