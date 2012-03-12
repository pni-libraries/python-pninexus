#File: nxgroup_ex1.py
import pni.nx.h5 as nx

nxfile = nx.create_file("nxgroup_ex1.h5",True,0)

#create groups
group = nxfile.create_group("data1")
group = group.create_group("dir")
group = nxfile.create_group("data2","NXentry")
group = nxfile.create_group("data3/detector/data","NXdata")

#open existing groups
group = nxfile.open("data1")
group = group.open("dir")

#open an existing group using [] operator
group = nxfile["/data3/detector/data"]

nxfile.close()
