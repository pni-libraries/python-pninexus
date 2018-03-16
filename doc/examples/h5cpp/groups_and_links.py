from __future__ import print_function
from pninexus import h5cpp
from pninexus.h5cpp import Path
from pninexus.h5cpp.node import Group,link
from pninexus.h5cpp.file import AccessFlags


h5file = h5cpp.file.create("groups_and_links.h5",AccessFlags.TRUNCATE)
root   = h5file.root()

def create_run_groups(base):
    
    sensors = Group(base,"sensors")
    Group(sensors,"temperature")
    p1 = Group(sensors,"pressure_001")
    p2 = Group(sensors,"pressure_002")
    Group(sensors,"pressure")
    
    link(p1,base,Path("sensors/pressure/pressure_0001"))
    link(p2,base,Path("sensors/pressure/pressure_0002"))
    Group(base,"logs")
    Group(base,"diagnostics")

#
# create a little group hierarchy
#
create_run_groups(Group(root,"run_0001"))
create_run_groups(Group(root,"run_0002"))
create_run_groups(Group(root,"run_0003"))

#
# create another link
#
link(Path("/run_0003"),root,Path("/current_run"))

#
# node iteration
#
print("SHOW ONLY RUN GROUPS")
print("====================")
for run in root.nodes:
    print(run.link.path)

print("\n")    
print("HOW THE ENTIRE TREE")
print("===================")
for run in root.nodes.recursive:
    print(run.link.path)
    
#
# link iteration
#
print("\n")
print("SHOW ONLY LINKS TO RUN GROUPS")
print("=============================")
for link in root.links:
    print("Link: {path} -> {type}".format(path=link.path,type=link.type()))
    
print("\n")
print("SHOW LINKS TO ALL GROUPS")
print("========================")
for link in root.links.recursive:
    print("Link: {path} -> {type}".format(path=link.path,type=link.type()))
