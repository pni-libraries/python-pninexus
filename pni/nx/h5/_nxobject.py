#module for NXObject specialization

from nxh5 import NXObject

from pni.utils.Array

class AttrManager(object):
    def __init__(self,o):
        self.__nxobject = o

    def __getattr__(self,name):
        pass

    def __setattr__(self,name,value):
        pass



