#setup script for libpninx-python package
import sys
import os
from distutils.core import setup
from distutils.extension import Extension
from distutils.fancy_getopt import FancyGetopt
from numpy.distutils.misc_util import get_numpy_include_dirs
from pkgconfig import package

cliopts =[]
cliopts.append(("h5libdir=",None,"HDF5 library path"))
cliopts.append(("h5incdir=",None,"HDF5 include path"))
cliopts.append(("h5libname=",None,"HDF5 library name"))
cliopts.append(("nxlibdir=",None,"PNI NX library path"))
cliopts.append(("nxincdir=",None,"PNI NX include path"))
cliopts.append(("utlibdir=",None,"PNI utilities library path"))
cliopts.append(("utincdir=",None,"PNI utilities include path"))
cliopts.append(("numpyincdir=",None,"Numpy include path"))
cliopts.append(("debug",None,"append debuging options"))

op = FancyGetopt(option_table=cliopts)
args,opts = op.getopt()

debug = False
for o,v in op.get_option_order():
    if o == "debug":
        debug = True

#add pniio libraries and flags
pniio        = package('pniio')
include_dirs = pniio.include_dirs
library_dirs = pniio.library_dirs
libraries    = pniio.libraries
libraries.append('boost_python')
include_dirs.extend(get_numpy_include_dirs())

extra_compile_args = ['-std=c++0x']
extra_compile_args.extend(pniio.compiler_flags)
if(debug):
    extra_compile_args.append('-O0')
    extra_compile_args.append('-g')


files = ["src/nx.cpp","src/NXWrapperHelpers.cpp","src/NXWrapperErrors.cpp"]

nxh5 = Extension("nxh5",files,
                 include_dirs = include_dirs,
                 library_dirs = library_dirs,
                 libraries = libraries,
                 extra_compile_args = extra_compile_args)

setup(name="libpniio-python",
        author="Eugen Wintersberger",
        author_email="eugen.wintersberger@desy.de",
        description="Python wrapper for libpniio",
        version = "0.9.4",
        ext_package="pni.io.nx.h5",
        ext_modules=[nxh5],
        packages = ["pni","pni.io","pni.io.nx","pni.io.nx.h5"],
        url="https://code.google.com/p/pni-libraries/",
        script_args = args
        )

