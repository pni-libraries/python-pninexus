from pninexus import h5cpp
from pninexus import nexus
import sys

#
# define a very simple template using Python string formatting 
# to create the basic file structure
#

file_struct =  \
"""
<group name="{scan_name}" type="NXentry">
    <group name="instrument" type="NXinstrument"/>
    <group name="data" type="NXdata"/>
    <group name="control" type="NXmonitor"/>
    <group name="sample" type="NXsample">
        <field name="name" type="string">{sample_name}</field>
        <field name="description" type="string">{sample_description}</field>
        <field name="material" type="string">Si + Ge</field>
    </group>
</group>
"""

mythen_detector = \
"""
<group name="mythen" type="NXdetector">
    <field name="layout" type="string">linear</field>
    <field name="x_pixel_size" type="float64" units="um">50</field>
    <field name="y_pixel_size" type="float64" units="mm">5</field>
</group>
"""

nxfile = nexus.create_file("xml.nxs",h5cpp.file.AccessFlags.TRUNCATE)
root   = nxfile.root()

file_instance = file_struct.format(scan_name="run_0001",
                                   sample_name="SMP01VA",
                                   sample_description="reference sample")

nexus.create_from_string(root,file_instance)

objects = nexus.get_objects(root,nexus.Path.from_string(":NXentry/:NXinstrument"))
if len(objects) == 1:
    instrument = objects[0]
else:
    print("Could not find instrument group!")
    sys.exit(1)
    
#
# add a detector
#
nexus.create_from_string(instrument,mythen_detector)
    



