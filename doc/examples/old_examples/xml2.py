import pni.io.nx.h5 as nx
from pni.io.nx import make_path

file_struct =  \
"""
<group name="entry" type="NXentry">
    <group name="instrument" type="NXinstrument"/>
    <group name="data" type="NXdata"/>
    <group name="control" type="NXmonitor"/>
    <group name="sample" type="NXsample">
        <field name="name" type="string">VK007</field>
        <field name="material" type="string">Si + Ge</field>
    </group>
</group>
"""

detector_struct = \
"""
<group name="mythen" type="NXdetector">
    <field name="layout" type="string">linear</field>
    <field name="x_pixel_size" type="float64" units="um">50</field>
    <field name="y_pixel_size" type="float64" units="mm">5</field>
</group>
"""


def write_if(obj):
    return (isinstance(obj,nx.nxfield) or 
           isinstance(obj,nx.nxattribute)) and obj.size ==1

f = nx.create_file("xml2.nxs",True)
root = f.root()

#create the basic object
nx.xml_to_nexus(file_struct,root,write_if)

instrument = nx.get_object(root,"/:NXentry/:NXinstrument")
nx.xml_to_nexus(detector_struct,instrument,write_if)

f.close()

