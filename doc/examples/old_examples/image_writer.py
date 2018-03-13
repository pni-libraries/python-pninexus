from __future__ import print_function
import numpy
import pni.io.nx.h5 as nexus

file_struct = \
"""
<group name="entry" type="NXentry">
    <group name="instrument" type="NXinstrument">
        <group name="detector" type="NXdetector"> 

        </group>
    </group>
</group>
"""

#create the NeXus file with the basic structure and obtain the detector
#group 
f = nexus.create_file("image_writer.nxs",True)
nexus.xml_to_nexus(file_struct,f.root())
detector = nexus.get_object(f.root(),"/:NXentry/:NXinstrument/:NXdetector")

#create the field where to store the data (initial size is 0)
nx = 1024
ny = 2048
deflate = nexus.deflate_filter()
deflate.rate = 2
data = detector.create_field("data","uint32",shape=[0,nx,ny],chunk=[1,nx,ny],
                             filter=deflate)

#write data to disk
print("start the measurement")
for index in range(20):
    data.grow(0,1)
    print("store frame {frame_index} ...".format(frame_index=index))
    data[index,...] = index

    f.flush()








