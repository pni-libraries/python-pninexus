##
## (c) Copyright 2014 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
##
## This file is part of python-pnicore.
##
## python-pnicore is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 2 of the License, or
## (at your option) any later version.
##
## python-pnicore is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with python-pnicore.  If not, see <http://www.gnu.org/licenses/>.
## ===========================================================================
##
## Created on: Oct 21, 2014
##     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
##

"""
Module implementing the HDF5 backend

"""

#from nxh5 import NXObject_NXObject as NXObject
from ._nxh5 import nxgroup
from ._nxh5 import nxfile
from ._nxh5 import nxfield
from ._nxh5 import nxattribute
from ._nxh5 import deflate_filter
from ._nxh5 import xml_to_nexus
from ._nxh5 import get_size
from ._nxh5 import get_name
from ._nxh5 import get_rank
from ._nxh5 import get_unit
from ._nxh5 import get_class
from ._nxh5 import get_object
from ._nxh5 import set_class
from ._nxh5 import set_unit
from ._nxh5 import get_path
from ._nxh5 import link


#import helper methods
from ._nxh5 import __create_file
from ._nxh5 import __open_file
from ._nxh5 import __create_files


def create_file(fname,overwrite=False):
    """create a new NeXus file

    This function creates a new NeXus file. All data will go o this file. If
    the file already exists the pni.core.FileError exception will be thrown. 
    In order to overwrite an existing file of equal name set the 
    overwrite flag to True.

    Args:
        fname (string):    the name of the file
        overwrite (bool):  overwrite flag (default False)

    Returns:
        A new instance of nxfile.

    Raises:
        pni.core.FileError in case of problems
    """
    return __create_file(fname,overwrite)

def create_files(fname,split_size,overwrite=False):
    """create a split file

    Create a new file which is splitted into subfiles of a given size. unlike
    for the create_file function the file name argument is a C-style 
    format string which includes a running index for the individual files. 

    A valid filename could look like this 
        data_file.%05i.nxs

    which would lead to file names like 
        data_file.00001.nxs
        data_file.00002.nxs
        data_file.00003.nxs


    Args: 
        fname (string):    Format string encoding the name of the files
        split_size (long): Size of the individual files
        overwrite (bool):  overwrite flag (default False)

    Returns: 
        A new instance of nxfile

    Raises:
        pni.core.FileError in case of problems
     
    """

    return __create_files(fname,split_size,overwrite)

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

def nxgroup_create_field(self,name,type,shape=None,chunk=None,filter=None):

    return self.__create_field(name,type,shape,chunk,filter)

def nxgroup_create_group(self,*args,**kwargs):
    if len(args) == 2:
        return self._create_group(args[0],args[1])
    elif len(args)==1:

        nxclass = None
        name    = None

        if "nxclass" in kwargs.keys():
            nxclass = kwargs["nxclass"]

        if ':' in args[0]:
            (name,nxclass) = args[0].split(":")
        else:
            name = args[0]
           
        if nxclass:
            return self._create_group(name,nxclass)
        else:
            return self._create_group(name)

    else:
        raise ValueError



nxgroup.create_field = nxgroup_create_field
nxgroup.create_group = nxgroup_create_group


