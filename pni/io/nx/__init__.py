##load objects from the pninx python C++ module

from .nxpath import nxpath
from .nxpath import match
from .nxpath import is_root_element
from .nxpath import is_absolute
from .nxpath import is_empty
from .nxpath import has_name
from .nxpath import has_class
from .nxpath import join
from .nxpath import make_relative
from .nxpath import make_path



def make_relative(parent,old):
    """Create a relative path

    Makes the path old a relative path to parent. For this function to succeed
    both paths must satisfy two conditions

    * old must be longer than parent
    * both must be absolut paths
    
    If any of these conditions is not satisfied a :py:class:`ValueError`
    exception will be thrown.

    .. code-block:: python

        parent = "/:NXentry/:NXinstrument"
        old    = "/:NXentry/:NXinstrument/:NXdetector/data"

        rel_path = make_relative(parent,old)
        print(rel_path)
        #output: :NXdetector/data

    :param str parent: the new root path 
    :param str old:  the original path which should be made relative to old
    :return: path refering to the same object as old but relative to parent
    :rtype: str
    :raises ValueError: old and parent do not satifisy the above conditions
    """

    return make_relative_(parent,old).__str__()

