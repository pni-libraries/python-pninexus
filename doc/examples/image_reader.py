from __future__ import print_function
import numpy
import pni.io.nx.h5 as nexus

#open the file 
det_path = "/:NXentry/:NXinstrument/:NXdetector/data"
f = nexus.open_file("image_writer.nxs")
data = nexus.get_object(f.root(),det_path)


for index in range(data.shape[0]):
    if numpy.equal(data[index,...],
                   index*numpy.ones((data.shape[1],data.shape[2]),dtype=data.dtype)).all():
        print("fame {frame_index} is ok".format(frame_index=index))

