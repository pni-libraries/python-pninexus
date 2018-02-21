from pni.io import nexus
from pni.io.nexus import match
from pni.io.nexus import is_absolute
from pni.io.nexus import join
from pni.io.nexus import make_relative as _make_relative

def get_name_and_base_class(*args):
    
    if len(args) == 1:
        (name,base_class) = args[0].split(":")
    else:
        name = args[0]
        base_class = args[1]

    return (name,base_class)

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

class nxpath(nexus.Path):
    
    
    def __init__(self,base_instance=None):
        if base_instance != None:
            super(nxpath,self).__init__(base_instance)
        else:
            super(nxpath,self).__init__()
    
    def __len__(self):
        return self.size
    
  
    def push_back(self,*args,**kwargs):
        """Append element to the end of the object section
       
        This method takes either two positional arguments
    
        .. code-block:: python
            
            path.push_back("entry","NXentry")
    
        Where the first one is the *name* and the optional second argument 
        the *base class* of the path element. 
    
        Alternatively there are two keyword arguments 
    
        .. code-block:: python
    
            path.push_back(name="entry",base_class="NXentry")
    
        which would have the same effect. Finally one can also use a single string 
        to describe a new element
    
        .. code-block:: python
    
            path.push_back("entry:NXentry")
            path.push_back(":NXinstrument")
    
        It must be noted that only a single element can be appended with 
        :py:meth:`push_back`.
        """
    
        (name,base_class) = get_name_and_base_class_from_args(*args,**kwargs)
        super(nxpath,self).push_back({"name":name,"base_class":base_class})


    def push_front(self,*args,**kwargs):
        """Prepends an element to a path
    
        This method works like :py:meth:`push_back` but prepends an element 
        in front of the first one. The arguments work the same as for 
        :py:meth:`push_back`.
    
        """
    
        (name,base_class) = get_name_and_base_class_from_args(*args,**kwargs)
    
        super(nxpath,self).push_front({"name":name,"base_class":base_class})
        
        

    def pop_front(self):
        """Remove first element from the object section of the path
        
        """
        if not len(self):
            raise IndexError("Object section of path is empty!")
    
        super(nxpath,self).pop_front()

    def pop_back(self):
        """Remove last element from the object section
        
        """
        if not len(self): 
            raise IndexError("Object section of path is empty!")
    
        super(nxpath,self).pop_back()
        
        
    def __add__(self,arg):
    
        if isinstance(arg,nxpath):
            return nxpath(base_instance=join(self,arg))
        elif isinstance(arg,str):
            return nxpath(base_instance=join(self,make_path(arg)))
        else:
            raise TypeError("Argument must be a string or an nxpath!")


    def __radd__(self,arg):
    
        if isinstance(arg,nxpath):
            return nxpath(base_instance=join(self,arg))
        elif isinstance(arg,str):
            return nxpath(base_instance=join(make_path(arg),self))
        else:
            raise TypeError("Argument must be a string or an nxpath!")
        
    def __iadd__(self,arg):
        
        return self.__add__(arg)
        
    def __str__(self):
        
        return nexus.Path.to_string(self)
    
    def __eq__(self,other):
        
        if isinstance(other,str):
            return self.__str__() == other
        else:
            return super(nxpath,self).__eq__(other)
        
    def __ne__(self,other):
        
        return not self.__eq__(other)
    

def is_root_element(path_element):
    
    if not isinstance(path_element,dict):
        return False
    
    if not (path_element.has_key("name") and path_element.has_key("base_class")):
        return False
    
    return path_element["name"]=="/" and path_element["base_class"]=="NXroot"

def is_empty(nexus_path):
    
    if nexus_path.size == 0:
        return True
    else:
        return False

def has_name(path_element):
    
    if not isinstance(path_element,dict):
        return False
    
    if path_element.has_key("name") and path_element["name"] != "":
        return True
    else:
        return False
    

def has_class(path_element):
    
    if not isinstance(path_element,dict):
        return False
    
    if path_element.has_key("base_class") and path_element["base_class"] != "":
        return True
    else:
        return False
    

def make_path(str_path):
    
    if isinstance(str_path,str):
        return nxpath(base_instance = nexus.Path.from_string(str_path))
    else:
        raise TypeError("Paths can only be constructed from strings!")

def match(path_a,path_b):
    
    if isinstance(path_a,str):
        path_a = nexus.Path.from_string(path_a)
        
    if isinstance(path_b,str):
        path_b = nexus.Path.from_string(path_b)
        
    return nexus.match(path_a,path_b)


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
 
    return nxpath(base_instance=_make_relative(parent,old))