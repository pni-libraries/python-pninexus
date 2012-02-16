

class DebControlEntry(object):
    def __init__(self):
        self.section = ""
        self.maintainer_name = ""
        self.maintainer_email = ""
        self.homepage = ""
        self.depends = ""

    def _get_maintainer(self):
        return self.maintainer_name+" <"+self.maintainer_email+">"

    maintainer=property(_get_maintainer)



class DebSourceControl(object):
    def __init__(self):
        self.source_name = ""
        self.priority = "extra" #default value extra
        self.standards_version = "3.9.1"
        self.homepage = ""

    def __str__(self):
        ostr =""
        ostr += "Source: "+self.source_name+"\n"
        ostr += "Priority: "+self.priority+"\n"
        ostr += "Maintainer: "+self.maintainer_name+" <"+self.maintainer_email+">\n"
        ostr += "Build-Depends: "+self.depends+"\n"
        ostr += "Standards-Version: "+self.standards_version+"\n"
        ostr += "Section: "+self.section+"\n"
        ostr += "Homepage: "+self.homepage+"\n"
        return ostr


class DebPackageControl(DebControl):
    def __init__(self):
        self.package_name = ""
        self.architecture = "any" #default value is "any"
        self.description = ""

    def __str__(self):
        ostr = ""
        ostr += "Package: "+self.package_name+"\n"
        ostr += "Section: "+self.section+"\n"
        ostr += "Architecture: "+self.architecture+"\n"
        ostr += "Depends: "+self.depends+"\n"
        ostr += "Description: "+self.description+"\n"
        return ostr 
        

class DebControl(list):
    def __init__(self):
        list.__init__(self)
    
    def write(self,fname):
        fid = open(fname)

        for ce in self:
            fid.write(ce.__str__())

        fid.close()





