from __future__ import print_function
import pni.io.nx.h5 as nx 

chunk = \
"""

"""

file_struct = \
"""
<group name="entry" type="NXentry">
    <group name="instrument" type="NXinstrument">
        <group name="mythen" type="NXdetector">
            <field name="layout" type="string">linear</field>
            <field name="x_pixel_size" type="float64" units="um">50</field>
            <field name="y_pixel_size" type="float64" units="mm">5</field>
            <field name="data" type="uint64" units="cps">
                <dimensions rank="2">
                    <dim index="1" value="42"/>
                    <dim index="2" value="2014"/>
                </dimensions>
            </field>

            <field name="depends_on" type="string"> transformation/tt</field>
            
            <group name="transformation" type="NXtransformation">
                <field name="tt" type="float64">
                    <dimensions rank="1">
                        <dim value="42" index="1"/>
                    </dimensions>

                    <attribute name="depends_on" type="string"> tth</attribute>
                    <attribute name="transformation_type" type="string">
                    rotation
                    </attribute>
                    <attribute name="vector" type="float64"> 
                        <dimensions rank="1">
                            <dim value="3" index="1"/>
                        </dimensions>
                    1 0 0 
                    </attribute>
                </field>
                
                <field name="tth" type="float64">
                    <dimensions rank="1">
                        <dim value="42" index="1"/>
                    </dimensions>

                    <attribute name="transformation_type" type="string">
                    rotation
                    </attribute>
                    <attribute name="vector" type="float64"> 
                        <dimensions rank="1">
                            <dim value="3" index="1"/>
                        </dimensions>
                    0 1 0 
                    </attribute>
                </field>
            </group>

        </group>
    </group>
    <group name="data" type="NXdata"/>
    <group name="control" type="NXmonitor"/>
    <group name="sample" type="NXsample">
        <field name="name" type="string">VK007</field>
        <field name="material" type="string">Si + Ge</field>

        <field name="depends_on" type="string"> transformation/phi</field>

        <group name="transformation" type="NXtransformation">
            <field name="phi" type="float32">
                <dimensions rank="1">
                    <dim value="42" index="1"/>
                </dimensions>

                <attribute name="depends_on" type="string"> chi </attribute>
                <attribute name="transformation_type" type="string"> rotation
                </attribute>

                <attribute name="vector" type="float32">
                    <dimensions rank="1"><dim value="3" index="1"/></dimensions>
                    0 1 0
                </attribute>
            </field>
            
            <field name="chi" type="float32">
                <dimensions rank="1">
                    <dim value="42" index="1"/>
                </dimensions>

                <attribute name="transformation_type" type="string"> rotation
                </attribute>

                <attribute name="vector" type="float32">
                    <dimensions rank="1"><dim value="3" index="1"/></dimensions>
                    0 0 1 
                </attribute>
            </field>
        </group>
    </group>
</group>
"""

def write_if(obj):
    return obj.size <= 3

f = nx.create_file("list_of_fields1.nxs",True)
nx.xml_to_nexus(file_struct,f.root(),write_if)

def scanned_field(obj):
    return isinstance(obj,nx.nxfield) and obj.shape[0]>1

scanned_fields = [obj for obj in f.root().recursive if scanned_field(obj)]

for field in scanned_fields:
    print(field.path)

f.close()
