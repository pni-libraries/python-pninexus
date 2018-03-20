from pninexus import h5cpp
from pninexus import nexus

nxfile = nexus.create_file("path_example.nxs",h5cpp.file.AccessFlags.TRUNCATE)
root   = nxfile.root()

nexus.create_from_file(root,"multidetector.xml")

#
# finding all the entries using a generic path
#
objects = nexus.get_objects(root,nexus.Path.from_string("/:NXentry"))
for object in objects:
    print(object.link.path)
    
#
# finding all detectors in the first scan
#
detectors = nexus.get_objects(root,nexus.Path.from_string("/scan_0001:NXentry/:NXinstrument/:NXdetector"))
for detector in detectors:
    print(detector.link.path)
    
#
# accessing a data field
#
data_path = nexus.Path.from_string("/scan_0001:NXentry/:NXinstrument/detector_001:NXdetector/data")
detector = nexus.get_objects(root,data_path)
print("Found {} detector".format(len(detector)))
detector = detector[0]
print(detector.link.path)

#
# accessing a single attribute
#
unit_path = nexus.Path.from_string("/scan_0001:NXentry/:NXinstrument/detector_001:NXdetector/data@units")
unit = nexus.get_objects(root,unit_path)[0]
print("detector unit: {}".format(unit.read()))
