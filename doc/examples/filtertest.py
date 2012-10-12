#!/usr/bin/env python

import pni.nx.h5 as nx

nxfile = nx.create_file("filtertest.h5",overwrite=True)
deflate = nx.NXDeflateFilter()
deflate.rate = 6
deflate.shuffle = True
field = nxfile.create_field("data","uint16",shape=(1024,1024),filter=deflate)
field = nxfile.create_field("data2","uint16",shape=(1024,1024),filter=5)
nxfile.close()
