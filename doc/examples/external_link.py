from __future__ import print_function
import pni.io.nx.h5 as nexus
import sys
import utilities as utils

master_file_struct = \
"""
<group name="entry" type="NXentry">
    <group name="instrument" type="NXinstrument">
        <group name="detector" type="NXdetector">
        </group>
    </group>

    <group name="sample" type="NXsample">
    </group>

    <group name="data" type="NXdata">
    </group>
</group>
"""

detector_file_struct = \
"""
<group name="entry" type="NXentry">
    <group name="instrument" type="NXinstrument">
        <group name="detector" type="NXdetector">
            <field name="data" type="uint32" units="cps">
                <dimensions rank="1">
                    <dim index="1" value="10"/>
                </dimensions>
                12 9 199 150 123 99 65 87 94 55
            </field>
        </group>
    </group>

</group>
"""

detector_file = nexus.create_file("detector.nxs",True)
root =detector_file.root()
nexus.xml_to_nexus(detector_file_struct,root,utils.write_everything)
root.close()
detector_file.close()

master_file = nexus.create_file("master_file.nxs",True)
root = master_file.root()
nexus.xml_to_nexus(master_file_struct,root)
try:
    dg = nexus.get_object(root,"/:NXentry/:NXinstrument/:NXdetector")
except KeyError:
    print("Could not find detector group in master file!")
    sys.exit(1)

nexus.link("detector.nxs://entry/instrument/detector/data",dg,"data")

data = nexus.get_object(root,"/:NXentry/:NXinstrument/:NXdetector/data")
print(data[...])


