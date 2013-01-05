#from nxh5 import NXObject_NXObject as NXObject
from nxh5 import NXObject
from nxh5 import NXGroup
from nxh5 import NXFile
from nxh5 import NXDeflateFilter
from nxh5 import NXField

from nxh5 import ShapeMissmatchError
from nxh5 import TypeError
from nxh5 import MemoryAllocationError
from nxh5 import IndexError

from nxh5 import NXFileError
from nxh5 import NXGroupError
from nxh5 import NXFieldError
from nxh5 import NXAttributeError
from nxh5 import NXFilterError


#import helper methods
from nxh5 import __create_file
from nxh5 import __open_file

def create_file(fname,overwrite=False,splitsize=0):
    """
    create_file(fname,overwrite=False,splitsize=0):
    Creates a new Nexus file. 

    arguments:
    fname ....... name of the file
    overwrite ... if true an existing file with the same name will be overwriten
    splitsize ... (feature not implemented - keep 0)

    return:
    Returns a new object of type NXFile.
    """
    return __create_file(fname,overwrite,splitsize)

def open_file(fname,readonly=True):
    """
    open_file(fname,readonly=True):
    Opens an existing Nexus file.

    arguments:
    fname ........ name of the file to open
    readonly ..... if True the file will be in read-only mode

    return:
    an instance of NXFile
    """
    return __open_file(fname,readonly)


