from __future__ import print_function
from pni.io import h5cpp
from pni.io import nexus
import time
import sys

file_structure = """
<group name="/" type="NXroot">
    <group name="{scan_name}" type="NXentry">
        <group name="instrument" type="NXinstrument">
        
            <group name="detector" type="NXdetector">
            
                <field name="data" type="uint64" units="cps">
                    <dimensions rank="1">
                        <dim value="0" index="1"/>
                    </dimensions>
                    <chunk rank="1">
                        <dim value="10000000" index="1"/>
                    </chunk>
                </field>
                
                <group name="transformations" type="NXtransformations">
                
                </group>
                
            </group>
            
        </group>
        
        <group name="sample" type="NXsample">
            <field name="name" type="string">{sample_name}</field>
            
            <group name="transformations" type="NXtransformations">
            
            </group>
        </group>
    </group>
</group>
"""

def create_file_property_lists():
   
    fcpl = h5cpp.property.FileCreationList()
    fapl = h5cpp.property.FileAccessList()
    fapl.library_version_bounds(h5cpp.property.LibVersion.LATEST,
                                h5cpp.property.LibVersion.LATEST)
    
    return (fcpl,fapl) 
    

def create_swmr_file(filename,scan_name="entry",sample_name="sample"):
    xml_struct = file_structure.format(sample_name=sample_name,
                                       scan_name=scan_name)
    
    fcpl,fapl = create_file_property_lists()
    f = nexus.create_file(filename,h5cpp.file.AccessFlags.TRUNCATE,fcpl,fapl)
    r = f.root()
    
    nexus.create_from_string(r,xml_struct)
    
    

def open_swmr_file(filename):
    
    fcpl,fapl = create_file_property_lists()
    
    f = nexus.open_file(filename,
                        h5cpp.file.AccessFlags.READWRITE | h5cpp.file.AccessFlags.SWMRWRITE,
                        fapl)
    
    return f
    

create_swmr_file("test.nxs")
nxfile = open_swmr_file("test.nxs")

detectors = nexus.get_objects(nxfile.root(),nexus.Path.from_string("/:NXentry/:NXinstrument/:NXdetector/data"))
if len(detectors)!=1:
    print("Too much or to less detectors found")
    sys.exit(1)
    
detector_data = detectors[0]

selection = h5cpp.dataspace.Hyperslab(offset=(detector_data.dataspace.size,),block=(1,))

for index in range(100):
    detector_data.extent(0,1)
    detector_data.write(index,selection)
    selection.offset(0,detector_data.dataspace.size)
    nxfile.flush()
    time.sleep(1)
    
print(detector_data.read())



