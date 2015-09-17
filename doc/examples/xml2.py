import pni.io.nx.h5 as nx 

file_struct =  \
"""
<group name="entry" type="NXentry">
    <group name="instrument" type="NXinstrument"/>
    <group name="data" type="NXdata"/>
    <group name="control" type="NXmonitor"/>
    <group name="sample" type="NXsample">
        <field name="name" type="string">VK007</field>
    </group>
</group>
"""

f = nx.create_file("xml1.nxs",True)
root = f.root()
nx.xml_to_nexus(file_struct,root)
f.close()

