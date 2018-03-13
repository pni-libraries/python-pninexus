from __future__ import print_function

from pninexus import h5cpp
from pninexus.h5cpp import Path
from pninexus.h5cpp.file import AccessFlags
from pninexus.h5cpp.node import Group,link


def create_external_data(filename):
    
    f = h5cpp.file.create(filename,AccessFlags.TRUNCATE)
    r = f.root()
    
    Group(r,"temperature_calibration")


create_external_data("sensor_calibration.h5")

h5file = h5cpp.file.create("sensor_data.h5",AccessFlags.TRUNCATE)
root   = h5file.root()

#
# create external link to existing group
#
link(target=Path("/temperature_calibration"),
     link_base = root,
     link_path = Path("/temperature_calibration"),
     target_file="sensor_calibration.h5")

#
# create external link to non-existing gruop
#
link(target=Path("/pressure_calibration"),
     link_base = root,
     link_path = Path("pressure_calibration"),
     target_file = "sensors_calibration.h5")

#
# show the result
#
for link in root.links:
    print("{path} -> {type} [{resolvable}]".format(path=link.path,
                                                   type=link.type(),
                                                   resolvable=link.is_resolvable))


