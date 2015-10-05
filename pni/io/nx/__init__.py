##load objects from the pninx python C++ module

from ._nx import nxpath
from ._nx import make_path
from ._nx import match
from ._nx import is_root_element
from ._nx import is_absolute
from ._nx import is_empty
from ._nx import has_name
from ._nx import has_class
from ._nx import join
from ._nx import make_relative_

#=============================================================================
def get_name_and_base_class(*args):
    
    if len(args) == 1:
        (name,base_class) = args[0].split(":")
    else:
        name = args[0]
        base_class = args[1]

    return (name,base_class)

#=============================================================================
def get_name_and_base_class_from_args(*args,**kwargs):

    if len(args)==1 and not len(kwargs):
        if args[0][0] == ':':
            return ("",args[0][1:])
        elif args[0].find(':') < 0:
            return (args[0],"")
        else:
            return(args[0].split(":"))
    elif len(args) == 2 and not len(kwargs):
        return (args[0],args[1])
    elif not len(args) and len(kwargs) == 2:
        if "name" in kwargs.keys() and "base_class" in kwargs.keys():
            return (kwargs["name"],kwargs["base_class"])
        else:
            raise KeyError("Wrong keyword arguments: must be 'name' and 'base_class'!")
    else:
        raise SyntaxError("Wrong number of positional or keyword arguments!")


#=============================================================================
def _nxpath_push_back(self,*args,**kwargs):

    (name,base_class) = get_name_and_base_class_from_args(*args,**kwargs)
    self._push_back({"name":name,"base_class":base_class})

#=============================================================================
def _nxpath_push_front(self,*args,**kwargs):

    (name,base_class) = get_name_and_base_class_from_args(*args,**kwargs)

    self._push_front({"name":name,"base_class":base_class})

#=============================================================================
def _nxpath_pop_front(self):
    if not len(self):
        raise IndexError("Object section of path is empty!")

    self._pop_front()

#=============================================================================
def _nxpath_pop_back(self):
    if not len(self): 
        raise IndexError("Object section of path is empty!")

    self._pop_back()

#=============================================================================
def _nxpath_add__(self,arg):

    if isinstance(arg,nxpath):
        return join(self,arg)
    elif isinstance(arg,str):
        return join(self,make_path(arg))
    else:
        raise TypeError("Argument must be a string or an nxpath!")

#=============================================================================
def _nxpath_radd__(self,arg):

    if isinstance(arg,nxpath):
        return join(self,arg)
    elif isinstance(arg,str):
        return join(make_path(arg),self)
    else:
        raise TypeError("Argument must be a string or an nxpath!")

nxpath.push_back = _nxpath_push_back
nxpath.push_front = _nxpath_push_front
nxpath.pop_front = _nxpath_pop_front
nxpath.pop_back  = _nxpath_pop_back
nxpath.__add__ = _nxpath_add__
nxpath.__iadd__ = _nxpath_add__
nxpath.__radd__ = _nxpath_radd__

def make_relative(parent,old):

    return make_relative_(parent,old).__str__()
