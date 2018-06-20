from __future__ import print_function
# from pni.io.nx import nxpath
from pni.io.nx import make_path
# from pni.io.nx import is_root_element
from pni.io.nx import has_class
from pni.io.nx import has_name

p = make_path("/:NXentry")
entry = p.back
print(entry)
print(has_class(entry))
print(entry)
print(has_name(entry))
print(entry)
