#debian package Builder for SCons

class DebBuilder(object):
    def __init__(self):
        self.package_list = []

    def appendPackage(self,package):
        self.package_list.append(package)

    def __call__(self,target,source,env):
        #here we need to run the debian package tools
        print "run dpkg-buildpackage ...."
