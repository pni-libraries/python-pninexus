from pni.io.nx import make_path
from pni.io.nx import is_root_element

a = make_path("/:NXentry")
print(is_root_element(a.front))
# print(a.front)
# print(a.front)
# print(is_root_element(a.front))
