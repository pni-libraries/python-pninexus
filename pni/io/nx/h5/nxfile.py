#
# (c) Copyright 2018 DESY               
#
# This file is part of python-pni.
#
# python-pni is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# python-pni is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with python-pni.  If not, see <http://www.gnu.org/licenses/>.
# ===========================================================================
#
# Created on: Feb 19, 2018
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#

from pni.io import nexus
from pni.io import h5cpp
from .nxgroup import nxgroup

class nxfile(h5cpp.file.File):
    
    def __init__(self,base_instance = None):
        if base_instance != None:
            super(nxfile,self).__init__(base_instance)
        else:
            super(nxfile,self).__init__()
            
    @property
    def filename(self):
        
        super(nxfile,self).path
        
    @property
    def readonly(self):
        
        if super(nxfile,self).intent == h5cpp.file.AccessFlags.READONLY:
            return True
        else:
            return False
        
    def root(self):
        
        return nxgroup(base_instance=super(nxfile,self).root())
    
        
        


def create_file(fname,overwrite=False):
    """create a new NeXus file

    This function creates a new NeXus file. All data will go o this file. If
    the file already exists the pni.core.FileError exception will be thrown. 
    In order to overwrite an existing file of equal name set the 
    overwrite flag to True.

    :param str fname:    the name of the file
    :param bool overwrite:  overwrite flag (default False)
    :return:  new file
    :rtype: instance of :py:class:`nxfile`
    :raises :py:exc:`pni.core.FileError`: in case of problems
    """
    
    flags = h5cpp.file.AccessFlags.EXCLUSIVE
    if overwrite:
        flags = h5cpp.file.AccessFlags.TRUNCATE
    
    return nxfile(base_instance = nexus.create_file(fname,flags))

def open_file(fname,readonly=True):
    """ Opens an existing Nexus file.

    *fname* can either be a simple path to a file or a C-style format string 
    as described for py::func:`create_files`. In the later case a series of
    NeXus files is opened.

    :param str fname: name of the file to open
    :param bool readonly: if True the file will be in read-only mode
    :return: new file
    :rtype: instance of :py:class:`nxfile`
    """
    
    flags = h5cpp.file.AccessFlags.READONLY
    if not readonly:
        flags = h5cpp.file.AccessFlags.READWRITE
    
    return nxfile(base_instance = nexus.open_file(fname,flags))


def create_files(fname,split_size,overwrite=False):
    """create a split file

    Create a new file which is splitted into subfiles of a given size. unlike
    for the create_file function the file name argument is a C-style 
    format string which includes a running index for the individual files. 

    A valid filename could look like this *data_file.%05i.nxs*
    which would lead to a series of files named

    .. code-block:: bash

        data_file.00001.nxs
        data_file.00002.nxs
        data_file.00003.nxs


    :param str fname:    Format string encoding the name of the files
    :param long split_size: Size of the individual files
    :param bool overwrite:  overwrite flag (default False)
    :return: a new instance of nxfile
    :rtype: instance of :py:class:`nxfile`
    :raises :py:exc:`pni.core.FileError`: in case of problems
    """

    raise NotImplementedError("Creating split files is currently not supported!")