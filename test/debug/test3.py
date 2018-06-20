from __future__ import print_function
import pni.io.nx.h5 as nx

f = nx.create_file("test.nxs", overwrite=True)
f.close()

f = nx.open_file("test.nxs")
print(f.readonly)
f.close()

f = nx.open_file("test.nxs", readonly=False)
print(f.readonly)
f.close()
