#setup script for libpniio-python package
import sys
import os
from distutils.core import setup
from distutils.extension import Extension
from distutils.fancy_getopt import FancyGetopt
from numpy.distutils.misc_util import get_numpy_include_dirs
from pkgconfig import package


#-------------------------------------------------------------------------
# add command line options
#-------------------------------------------------------------------------
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

#-------------------------------------------------------------------------
# parse command line options
#-------------------------------------------------------------------------
op = FancyGetopt(option_table=cliopts)
args,opts = op.getopt()

debug = False
for o,v in op.get_option_order():
    if o == "debug":
        debug = True

#-------------------------------------------------------------------------
# set up compiler and linker paths and flags
#-------------------------------------------------------------------------
core_path = '/home/eugen/Development/DESY/lib/python2.7/site-packages/pni/core'
#add pniio libraries and flags
pniio        = package('pniio')
include_dirs = pniio.include_dirs
library_dirs = pniio.library_dirs
library_dirs.append(core_path)
libraries    = pniio.libraries
libraries.append('boost_python')
libraries.append(':_core.so')
include_dirs.extend(get_numpy_include_dirs())


extra_compile_args = ['-std=c++11','-Wall','-Wextra',
                      '-fdiagnostics-show-option']
extra_compile_args.extend(pniio.compiler_flags)
if(debug):
    extra_compile_args.append('-O0')
    extra_compile_args.append('-g')


#-------------------------------------------------------------------------
# the io.nx.h5 extension
#-------------------------------------------------------------------------
nxh5_files = ["src/_nxh5.cpp",
         "src/utils.cpp",
         ]

nxh5 = Extension("io.nx.h5._nxh5",nxh5_files,
                 include_dirs = include_dirs,
                 library_dirs = library_dirs,
                 libraries = libraries,
                 runtime_library_dirs=[core_path],
                 language="c++",
                 extra_compile_args = extra_compile_args)

#-------------------------------------------------------------------------
# the io extension
#-------------------------------------------------------------------------
io_files = ["src/_io.cpp","src/errors.cpp"]
io = Extension("io._io",io_files,
                 include_dirs = include_dirs,
                 library_dirs = library_dirs,
                 libraries = libraries,
                 runtime_library_dirs=[core_path],
                 language="c++",
                 extra_compile_args = extra_compile_args)

#-------------------------------------------------------------------------
# the global setup for the package
#-------------------------------------------------------------------------
setup(name="libpniio-python",
        author="Eugen Wintersberger",
        author_email="eugen.wintersberger@desy.de",
        description="Python wrapper for libpniio",
        version = "1.0.0",
        ext_package="pni",
        ext_modules=[nxh5,io],
        packages = ["pni","pni.io","pni.io.nx","pni.io.nx.h5"],
        url="https://code.google.com/p/pni-libraries/",
        script_args = args
        )

