from __future__ import print_function
from pninexus import h5cpp
from pninexus.h5cpp.node import Group,get_node
from pninexus.h5cpp.file import AccessFlags
from pninexus.h5cpp import Path
from pninexus.h5cpp.property import LinkCreationList,GroupCreationList,GroupAccessList 

h5file = h5cpp.file.create("intermediate_groups.h5",AccessFlags.TRUNCATE)
root   = h5file.root()

lcpl = LinkCreationList()
lcpl.intermediate_group_creation = True

Group(root,"run_0001/sensors/temperature",lcpl=lcpl)

for node in root.nodes.recursive:
    print(node.link.path)
    
    
temperature = get_node(root,Path("run_0001/sensors/temperature"))
print("Got: {}".format(temperature.link.path))
