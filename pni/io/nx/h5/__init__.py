#from nxh5 import NXObject_NXObject as NXObject
from ._nxh5 import nxgroup
from ._nxh5 import nxfile
from ._nxh5 import nxfield
from ._nxh5 import nxattribute
from ._nxh5 import deflate_filter


#import helper methods
from ._nxh5 import __create_file
from ._nxh5 import __open_file

def create_file(fname,overwrite=False):
    """
    create_file(fname,overwrite=False):
    Creates a new Nexus file. 

    arguments:
    fname ....... name of the file
    overwrite ... if true an existing file with the same name will be overwriten

    return:
    Returns a new object of type NXFile.
    """
    return __create_file(fname,overwrite)

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


