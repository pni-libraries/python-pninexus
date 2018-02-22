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

from pni.io import nexus

from .nxfile import nxfile
from .nxfile import create_file
from .nxfile import create_files
from .nxfile import open_file

from .nxgroup import nxgroup
#from nxh5 import NXObject_NXObject as NXObject
from .nxfield import nxfield
from .nxattribute import nxattribute
#from ._nxh5 import deflate_filter
from .algorithms import get_name
from .algorithms import get_rank
from .algorithms import get_unit
from .algorithms import get_class
from .algorithms import get_object
from .algorithms import set_class
from .algorithms import set_unit
#from ._nxh5 import _get_path_from_attribute
#from ._nxh5 import _get_path_from_dataset
#from ._nxh5 import _get_path_from_group
#from ._nxh5 import _get_path_from_link
from .nxlink import link
from .nxlink import nxlink
#from ._nxh5 import nxlink_status
#from ._nxh5 import nxlink_type
#from ._nxh5 import get_links_recursive
#from ._nxh5 import get_links


#import helper methods
#from ._nxh5 import __create_file
#from ._nxh5 import __open_file
#from ._nxh5 import __create_files

#add the path property to the nxlink class
#nxlink.path = property(lambda self: get_path(self.parent)+"/"+self.name)

def get_path(object):
    """ Return the NeXus path of an object
    
    Return the full NeXus path of an object. The object must be either an 
    instance of :py:class:`nxfield`, :py:class:`nxgroup`, :py:class:`nxattribute`
    or :py:class:`nxlink`.
    
    :param object object: instance for which to determine the path
    :return: full NeXus path
    :rtype: str
    """
    if isinstance(nexus_object,nxattribute):
        return _get_path_from_attribute(nexus_object);
    elif isinstance(nexus_object,nxgroup):
        return _get_path_from_group(nexus_object);
    elif isinstance(nexus_object,nxfield):
        return _get_path_from_dataset(nexus_object);
    else:
        raise TypeError("unknown NeXus object type")

def get_size(object):
    """Returns the size of an object
    
    The semantics of size depends on the object. In the case of a group 
    the number of children is returned. For attributes and fields the 
    number of elements. The object must be either an instance of 
    :py:class:`nxfield`, :py:class:`nxattribute` or :py:class:`nxgroup`.

    :param object object: object for which to determine the size
    :return: number of elements or children
    :rtype: long
    """
    
    if isinstance(object,(nxattribute,nxfield,nxgroup)):
        return object.size()
    else:
        raise TypeError("Object must be an instance of attribute, field, or group!")



def nxgroup_create_field(self,name,type,shape=None,chunk=None,filter=None):
    """Create a new field
    
    Create a new field below this group. 

    :param str name: the name of the new field
    :param str type: a numpy type string determining the data type of the field
    :param tuple shape: number of elements along each dimension
    :param tuple chunk: the chunk shape of the field
    :param filter: a filter object to compress the data
    :return: instance of a new field
    :rtype: instance of :py:class:`nxfield`
    """

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

def nxgroup_names(self):
    
    return [x.name for x in self]
        


#nxgroup.create_field = nxgroup_create_field
#nxgroup.create_group = nxgroup_create_group
#nxgroup.names = nxgroup_names


def xml_to_nexus(xml_data,parent):
    
    
    return nexus.create_from_string(parent,xml_data)


