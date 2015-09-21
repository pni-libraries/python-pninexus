import pni.io.nx.h5 as nx

file_struct =  \
"""
<group name="entry" type="NXentry">
    <group name="instrument" type="NXinstrument">
        <group name="mythen" type="NXdetector">
            <field name="layout" type="string">linear</field>
            <field name="x_pixel_size" type="float64" units="um">50</field>
            <field name="y_pixel_size" type="float64" units="mm">5</field>
        </group>
    </group>
    <group name="data" type="NXdata"/>
    <group name="control" type="NXmonitor"/>
    <group name="sample" type="NXsample">
        <field name="name" type="string">VK007</field>
        <field name="material" type="string">Si + Ge</field>
    </group>
</group>
"""

f = nx.create_file("rec_group_iter.nxs",True)
root = f.root()

def write_if(obj):
    return (isinstance(obj,nx.nxfield) or 
           isinstance(obj,nx.nxattribute)) and obj.size ==1

nx.xml_to_nexus(file_struct,root,write_if)

for x in root.recursive:
    print nx.get_path(x)
    if isinstance(x,nx.nxfield):
        print "data: ",x[...]


f.close()
