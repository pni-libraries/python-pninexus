#setup script for python-pnicore
import sys
import os
from distutils.core import setup
from distutils.extension import Extension
from distutils.fancy_getopt import FancyGetopt
from numpy.distutils.misc_util import get_numpy_include_dirs
from pkgconfig import package

#-----------------------------------------------------------------------------
# manage command line arguments
#-----------------------------------------------------------------------------
cliopts =[]
cliopts.append(("debug",None,"append debuging options"))

op = FancyGetopt(option_table=cliopts)
args,opts = op.getopt()

debug = False
for o,v in op.get_option_order():
    if o == "debug":
        debug = True

#-----------------------------------------------------------------------------
# load pnicore configuration with pkg-config
#-----------------------------------------------------------------------------
pnicore        = package('pnicore')

#add the configuration to libraries, include directories and library 
#directories
include_dirs = pnicore.include_dirs
library_dirs = pnicore.library_dirs
libraries    = pnicore.libraries

#-----------------------------------------------------------------------------
# set compiler options
#-----------------------------------------------------------------------------
extra_compile_args = ['-std=c++11','-Wall','-Wextra',
                      '-fdiagnostics-show-option']
extra_compile_args.extend(pnicore.compiler_flags)
if(debug):
    extra_compile_args.append('-O0')
    extra_compile_args.append('-g')

#-----------------------------------------------------------------------------
# list of files for the pnicore extensions
#-----------------------------------------------------------------------------
files = ["src/bool_converter.cpp",
         "src/numpy_scalar_converter.cpp",
         "src/numpy_utils.cpp",
         "src/nxwrapper_errors.cpp",
         "src/nxwrapper_utils.cpp",
         "src/pnicore.cpp"]

#-----------------------------------------------------------------------------
# setup for the pnicore extension
#-----------------------------------------------------------------------------
pnicore_ext = Extension("pnicore",files,
                        include_dirs = include_dirs,
                        library_dirs = library_dirs,
                        libraries = libraries,
                        language="c++",
                        extra_compile_args = extra_compile_args)

#-----------------------------------------------------------------------------
# setup for the pnicore package
#-----------------------------------------------------------------------------
setup(name="libpnicore-python",
        author="Eugen Wintersberger",
        author_email="eugen.wintersberger@desy.de",
        description="Python wrapper for libpnicore",
        version = "1.0.0",
        ext_package="pni.core",
        ext_modules=[pnicore_ext],
        packages = ["pni","pni.core"],
        url="https://code.google.com/p/pni-libraries/",
        script_args = args
        )

