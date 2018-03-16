from __future__ import print_function
import pni.io.nx.h5 as nexus
import numpy
import utilities as utils

file_struct = \
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
            <group name="transformations" type="NXtransformations">
                <field name="tt" type="float64" units="degree">
                    <dimensions rank="1">
                        <dim index="1" value="10"/>
                    </dimensions>
                2.0 4.0 6.0 8.0 10. 12. 14. 16. 18. 20.
                </field>
            </group>
        </group>
    </group>

    <group name="sample" type="NXsample">
        <group name="transformations" type="NXtransformations">
            <field name="om" type="float64" units="degree">
                <dimensions rank="1">
                    <dim index="1" value="10"/>
                </dimensions>
            1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.
            </field>
        </group>
    </group>

    <group name="data" type="NXdata">
    </group>
</group>
"""




f = nexus.create_file("internal_link.nxs",True)
r = f.root()
nexus.xml_to_nexus(file_struct,r,utils.write_everything)

nxdata = nexus.get_object(r,"/:NXentry/:NXdata")

#link an object
data = nexus.get_object(r,"/:NXentry/:NXinstrument/:NXdetector/data")
nexus.link(data,nxdata,"data")

#link to a path
nexus.link("/entry/sample/transformations/om",nxdata,"om")
nexus.link("/entry/detector/transformations/tt",nxdata,"tt")

#finalize the nxdata structure for easy plotting
nxdata.attributes.create("signal","string")[...]="data"
nxdata.attributes.create("axes","string",shape=(2,))[...] = \
numpy.array(["om","tt"])

nxdata.attributes.create("tt_indices","uint32")[...] = 0
nxdata.attributes.create("om_indices","uint32")[...] = 0






