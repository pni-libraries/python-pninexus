#File: nxfile_ex1.cpp

import nx.pni.h5 as nx

nxfile = nx.create_file("file_ex1.h5",True,0)
nxfile.close()

