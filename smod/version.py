#module contains classes to represent program and library versions as well 
#as classes to determine these versions. 

import subprocess
from distutils import sysconfig
import os

class ProgramVersion:
    """
    ProgramVersion
    This class represents the version number of a program. The version 
    number is considered to be of the form:
    
    major.minor.release
    
    where all three are integers. 
    """
    def __init__(self,major,minor,release):    
        self.major = major
        self.minor = minor
        self.release = release
        
    def __str__(self):        
        ostr = "%i." %(self.major)
        ostr += "%i." %(self.minor)
        ostr += "%i" %(self.release)
        
        return ostr
    
    def __eq__(self,other):
        if not self.__ne__(other): return True
        
        return False
    
    def __ne__(self,other):
        if self.major != other.major: return True
        if self.minor != other.minor: return True
        if self.release != other.release: return True
        
        return False
    
    def __lt__(self,other):
        if self.major < other.major: return True
        if self.minor < other.minor: return True
        if self.release < other.release: return True
        
        return False
    
    def __le__(self,other):
        if self.__lt__(other) or self.__eq__(other): return True
        
        return False
    
    def __gt__(self,other):
        if self.major > other.major: return True
        if self.minor > other.minor: return True
        if self.release > other.release: return True
        
        return False
    
    def __ge__(self,other):
        if self.__gt__(other) or self.__eq__(other): return True
        
        return False
    
    

class LibraryVersion(ProgramVersion):
    """
    LibraryVersion
    Represents the version of a library. This is basically the same as
    ProgramVersion. However, it has an additional integer attribute 
    soversion representing the ABI version of the library.
    """
    def __init__(self,major,minor,release,soversion):
        ProgramVersion.__init__(self,major,minor,release)
        self.soversion = soversion
        
    def __str__(self):
        ostr = ProgramVersion.__str__()
        ostr += ".%i" %(self.soversion)
        
        return ostr
    
class VersionParser:
    """
    VersionParser
    is the base class for all parsers for program an library versions. 
    It calls a program with a couple of arguments which should return 
    a string that is already or contains the program or library version. 
    In particular the case of libraries this can be a bit more complex.
    """
    def __init__(self,program,options):
        self.program = program
        self.options = options
    
    def parse(self,prog=None,opts=None):
        runprog = self.program
        runopts = self.options
        
        if prog: runprog = prog
        if opts: runopts = opts
            
        proc = subprocess.Popen([runprog,runopts],stdout=subprocess.PIPE)
        vstr = proc.communicate()[0]
        
        return vstr
    
class GCCVersionParser(VersionParser):
    """
    GCCVersionParser
    is a descendant of VersionParser and suitable fo get the program version
    of the gcc installation on a particular system. 
    """
    def __init__(self):
        VersionParser.__init__(self,"gcc","-dumpversion")
        
    def parse(self,prog=None,opts=None):
        vstr = VersionParser.parse(self,prog,opts)
        vlist = vstr.split(".")
        
        major = 0
        minor = 0
        release = 0
        
        for i in range(len(vlist)):
            if i==0: major = int(vlist[i])
            if i==1: minor = int(vlist[i])
            if i==2: release = int(vlist[i])
            
        return ProgramVersion(major,minor,release)
    
class DoxyVersionParser(VersionParser):
    def __init__(self): 
        VersionParser.__init__(self,"doxygen","")
    
    def parse(self,prog=None,opts=None):
        vstr = VersionParser.parse(self,prog,opts)
        version_line = vstr.split("\n")[0]
        version_str = version_line.split(" ")[-1]
        version_list = version_str.split(".")
        
        major = 0
        minor = 0
        release = 0
        
        for i in range(len(version_list)):
            if i==0: major = int(version_list[i])
            if i==1: minor = int(version_list[i])
            if i==2: release = int(version_list[i])
            
        return ProgramVersion(major,minor,release)
        
class PythonVersionParser(VersionParser):
    def __init__(self):
        VersionParser.__init__(self,"","")

    def parse(self,prog=None,opts=None):
        vstr = sysconfig.get_python_version()

        vlist = vstr.split(".")
        major = 0
        minor = 0
        release = 0

        for i in range(len(vlist)):
            if i==0: major = int(vlist[i])
            if i==1: minor = int(vlist[i])
            if i==2: release = int(vlist[i])

        return ProgramVersion(major,minor,release)
