#master build script for the libpninx python bindings


import os.path as path


#================define some variables=====================
var = Variables()
#installation paths for boost
var.Add(PathVariable(
        "BOOSTLIBDIR","BOOST library installation directory","/usr/lib"))
var.Add(PathVariable(
        "BOOSTINCDIR","BOOST header installation directory","/usr/include"))

#installation paths for libpniutils
var.Add(PathVariable(
        "PNIUTILSLIBDIR","PNI utils library installation directory","/usr/lib"))
var.Add(PathVariable(
        "PNIUTILSINCDIR","PNI utils header installation directory",
        "/usr/include"))

#installation paths for libpninx
var.Add(PathVariable(
        "PNINXLIBDIR","PNI nexus library installation directory","/usr/lib"))
var.Add(PathVariable(
        "PNINXINCDIR","PNI nexus header installation directory",
        "/usr/include"))

#compiler to use
var.Add("CXX","C++ compiler to use","g++")

#====================create build environment==================
env = Environment(variables=var)

#set the compiler used for the build
env.Replace(CXX = env["CXX"])



SConscript["src/SConscript"]

