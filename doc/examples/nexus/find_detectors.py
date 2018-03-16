#
# examle showing how to work with primitive search predicates
#
import sys
from pninexus import h5cpp
from pninexus import nexus


nxfile = nexus.create_file("find_detectors.nxs",h5cpp.file.AccessFlags.TRUNCATE)
root   = nxfile.root()
nexus.create_from_file(root,"multidetector.xml")

#
# assemble list of entries
#
is_entry = nexus.IsEntry()
is_detector = nexus.IsDetector()
entries = [group for group in root.nodes if is_entry(group)]

for entry in entries:
    detectors = [group for group in entry.nodes.recursive if is_detector(group)]
    
    print("Entry: {path} found {detectors} detectors".format(path = entry.link.path,
                                                             detectors = len(detectors)))
    for detector in detectors:
        print("\t"+str(detector.link.path))
