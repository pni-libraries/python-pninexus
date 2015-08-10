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


