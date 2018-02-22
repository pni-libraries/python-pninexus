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

from pni.io import h5cpp
from pni.io import nexus

from .nxfield import nxfield
from .nxgroup import nxgroup
from .nxattribute import nxattribute
from .nxlink import nxlink
from ..nxpath import nxpath


def get_name(object):
    pass

def get_rank(object):
    
    if not isinstance(object,(h5cpp.node.Dataset,nxfield,h5cpp.attribute.Attribute,nxattribute)):
        raise TypeError("`object` must be an instance of `Dataset`, `Attribute`, `nxfield` or `nxattribute`!")

def get_unit(field):
    
    if not isinstance(field,(h5cpp.node.Datatype,nxfield)):
        raise TypeError("`field` must be an instance of `nxfield` or `Dataset`!")
    
    if field.attributes.exists("units"):
        return field.attributs["units"].read()
    else:
        return str()
    
    

def get_class(group):
    
    if not isinstance(group,(h5cpp.node.Group,nxgroup)):
        raise TypeError("`group` must be an instance of `Group` or `nxgroup`!")
    
    if not group.attributes.exists("NX_class"):
        return ""
    
    return group.attributes["NX_class"].read()


def __convert_to_nx_object(object):
    """Convert an nx* object
    
    Internal utility function converting a new API object to an old style nx*
    instance. The new interface works entirely with h5cpp objects. To keep the 
    old interface theses objects must be put into their corresponding adapter 
    instances. 
    
    In order to succeed `object` must be an instance of 
    
    * :py:class:`pni.io.h5cpp.node.Group`
    * :py:class:`pni.io.h5cpp.node.Dataset`
    * :py:class:`pni.io.h5cpp.attribute.Attribute` 
    
    :param object: object to convert 
    :raises TypeError: if the type of object is not supported
    
    """
    if isinstance(object,h5cpp.node.Group):
        return nxgroup(base_instance = object)
    elif isinstance(object,h5cpp.node.Dataset):
        return nxfield(base_instance = object)
    elif isinstance(object,h5cpp.attribute.Attribute):
        return nxattribute(base_instance = object)
    else:
        raise TypeError("Return type not supported")
    

     

def get_object(parent,path):
    """get an object from a group
    
    `path` must be either an instance of 
    
    * `h5cpp.Path`
    * `str`
    * or `nexus.Path`
    
    :param Group/nxgroup parent: the parent at which to start object search
    :param path: the path to use for the search
    """
    
    def __to_single_object(objects):
        #
        # handling error situations:
        #
        # -> if more than one object was returned the path is ambiguous and we 
        #    throw an exception
        # -> if no object matches the path we throw too - this should maybe be 
        #    changed to something else.
        #
        if len(objects) > 1:
            message = "Path [{}] references more than one object!\n".format(path)
            
            for object in objects:
                message += "{}\n".format(object.link.path)
                
            raise ValueError(message)
        elif len(objects)==0:
            parent_path = nexus.get_path(parent)
            message = "Path [{}] references no object below [{}]".format(path,parent_path)
            raise ValueError(message)
        
        return objects[0]
        
    
    if isinstance(path,str):
        path = nexus.Path.from_string(path)
    elif isinstance(path,h5cpp.Path):
        path = nexus.Path(path)
    elif isinstance(path,(nxpath,nexus.Path)):
        pass
    else:
        raise TypeError("Path is of inapproprirate type!")
    
    #
    # if we sit on the root group and the root group is requested we have to
    # return the root group 
    #
    if parent.link.path.name == "." and path.__str__() == "/":
        nodes = [parent]
    else:
        try:
            nodes = nexus.get_objects(parent,path)
        except RuntimeError, error:
            print(error)
            #
            # if we fail to acquire any objects we try to obtain a link to the
            # very last element (which might be a dangling link)
            #
            last_element_name = path.back["name"]
            
            if last_element_name == "":
                message = "Could not find [{path}] - and last name does not have a name!".format(path=str(path))
                raise ValueError(message)
            
            #
            # remove the last element from the current path and try again
            # - if we're lucky there is at list a link of that name 
            #
            print(path.size)
            path.pop_back()
            print(path.size)
            print("LAST RESORT: ",str(path),type(path))
            nodes = nexus.get_objects(parent,nexus.Path.from_string("/:NXentry/:NXinstrument"))
            node = __to_single_object(nodes)
            
            if node.links.exists(last_element_name):
                print("LAST ELEMENT NAME: ",last_element_name)
                return nxlink(base_instance = node.links[last_element_name])
            
            else:
                raise RuntimeError("Error finding object")
    
    
    
    node = __to_single_object(nodes)
    
    return __convert_to_nx_object(node)
    
    
    

def set_class(group):
    pass

def set_unit(field):
    pass
    