##load objects from the pninx python C++ module

from ._nx import nxpath
from ._nx import make_path
from ._nx import match
from ._nx import is_root_element
from ._nx import is_absolute
from ._nx import is_empty
from ._nx import has_name
from ._nx import has_class

def _nxpath_append(self,name="",base_class=""):

    if not name and not base_class:
        raise value_error,"name and base_class argument must not both be empty!"

    self._append({"name":name,"base_class":base_class})

def _nxpath_prepend(self,name="",base_class=""):

    if not name and not base_class:
        raise value_error,"name and base_class argument must not both be empty!"

    self._prepend({"name":name,"base_class":base_class})

def _nxpath_pop_front(self):
    if not len(self):
        raise IndexError,"Object section of path is empty!"

    self._pop_front()

def _nxpath_pop_back(self):
    if not len(self): 
        raise IndexError,"Object section of path is empty!"

    self._pop_back()

nxpath.append = _nxpath_append
nxpath.prepend = _nxpath_prepend
nxpath.pop_front = _nxpath_pop_front
nxpath.pop_back  = _nxpath_pop_back
